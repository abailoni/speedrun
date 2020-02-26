from contextlib import contextmanager
import torch
from torch.utils.data import DataLoader
import os
from .py_utils import locate, get_single_key_value_pair, create_instance
from .log_anywhere import register_logger, log_scalar
from .tensorboard import TensorboardMixin
from copy import deepcopy

try:
    import inferno
    from inferno.utils import python_utils as pyu
    from inferno.io.core.base import IndexSpec
except ImportError:
    inferno = None
try:
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
except ImportError:
    TensorboardLogger = None

try:
    from inferno.trainers.callbacks.base import Callback
except ImportError:
    Callback = object

try:
    from firelight.inferno_callback import get_visualization_callback as firelight_visualizer
except ImportError:
    firelight_visualizer = None

# logger to sent images to the trainer to be visualized by firelight from anywhere


class FirelightLogger(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def log_image(self, tag, value):
        self.trainer.update_state(tag, value.detach().cpu())


class IncreaseStepCallback(Callback):
    def __init__(self, experiment):
        super(IncreaseStepCallback, self).__init__()
        self.experiment = experiment

    def end_of_training_iteration(self, **_):
        self.experiment.next_step()
        log_scalar('step', self.experiment.step)

    def __getstate__(self):
        return {}


class ParsingMixin(object):
    """
    The ParsingMixin provides a convenient way to create
    model, criterion metric and data loaders from
    You may overwrite the build methods in the derived experiment classes
    to define the model, criterion and metric that are harder to instantiate
    (e.g. by requiring specially constructed input objects)
    Note, that these methods should return the object in question,
    which makes it possible to use the parent objects
    """

    MODEL_LOCATIONS = []
    DATASET_LOCATIONS = []
    CRITERION_LOCATIONS = []
    METRIC_LOCATIONS = []

    def build_model(self, model_dict=None):
        if model_dict is None:
            model_dict = self.get('model')
        elif isinstance(model_dict, str):
            model_dict = self.get(model_dict)
        
        model_path = model_dict[next(iter(model_dict.keys()))].pop('loadfrom', None)
        model = create_instance(model_dict, self.MODEL_LOCATIONS)
        
        if model_path is not None:
            print(f"loading model from {model_path}")
            loaded_model = torch.load(model_path)
            loaded_model = loaded_model if isinstance(loaded_model, torch.nn.Module) else loaded_model["_model"]
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict)
        return model

    @property
    def model(self):
        # Build model if it doesn't exist
        if not hasattr(self, '_model'):
            # noinspection PyAttributeOutsideInit
            self._model = self.build_model()
            assert self._model is not None
        return self._model

    def build_train_loader(self):
        loader_kwargs = self.get('loader')
        dataset = create_instance(loader_kwargs['dataset'], self.DATASET_LOCATIONS)
        loader_kwargs['dataset'] = dataset
        return DataLoader(**loader_kwargs)

    # overwrite this function to define validation loader
    def build_val_loader(self):
        loader_kwargs = self.get('val_loader')
        dataset = create_instance(loader_kwargs['dataset'], self.DATASET_LOCATIONS)
        loader_kwargs['dataset'] = dataset

        return DataLoader(**loader_kwargs)

    # overwrite this function to define infer loader
    def build_infer_loader(self):
        loader_kwargs = self.get('infer_loader')
        dataset = create_instance(loader_kwargs['dataset'], self.DATASET_LOCATIONS)
        loader_kwargs['dataset'] = dataset

        return DataLoader(**loader_kwargs)


    def build_criterion(self):
        return create_instance(self.get('criterion'), self.CRITERION_LOCATIONS)

    def build_metric(self):
        metric = self.get('metric')
        if metric is not None:
            metric = create_instance(metric, self.METRIC_LOCATIONS)
        return metric

    @property
    def criterion(self):
        if not hasattr(self, '_criterion'):
            # noinspection PyAttributeOutsideInit
            self._criterion = self.build_criterion()
        return self._criterion

    @property
    def metric(self):
        if not hasattr(self, '_metric'):
            # noinspection PyAttributeOutsideInit
            self._metric = self.build_metric()
        return self._metric


class InfernoMixin(ParsingMixin):

    TRANSFORM_LOCATIONS = []

    @property
    def tagscope(self):
        if not hasattr(self, '_tagscope'):
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''
        return self._tagscope

    @contextmanager
    def set_tagscope(self, name):
        try:
            self._tagscope = name
            yield
        finally:
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''

    def get_full_tag(self, tag):
        if self.tagscope:
            return "{}/{}".format(self.tagscope, tag)
        else:
            return tag

    @property
    def device(self):
        if self._device is None:
            # noinspection PyAttributeOutsideInit
            self._device = torch.device(self.get('device'))
        return self._device

    @property
    def trainer(self):
        """
        inferno trainer. Will be constructed on first use.
        """
        if inferno is None:
            raise ModuleNotFoundError("InfernoMixin requires inferno. You can "
                                      "install it with `pip install in "
                                      "pip install inferno-pytorch`")
        # Build trainer if it doesn't exist
        if not hasattr(self, '_trainer'):
            # noinspection PyAttributeOutsideInit
            self._trainer = inferno.trainers.basic.Trainer(self.model)\
                                   .save_to_directory(self.experiment_directory)

            # call all defined bind functions
            for fname in dir(self):
                if fname.startswith('inferno_build_'):
                    getattr(self, fname)()

            # add callback to increase step counter
            # noinspection PyUnresolvedReferences
            self._trainer.register_callback(IncreaseStepCallback(self))

            self._trainer.to(self.device)

        return self._trainer

    def inferno_build_criterion(self):
        self._trainer.build_criterion(self.criterion)

    def inferno_build_metric(self):
        if self.metric is not None:
            self._trainer.build_metric(self.metric)
        else:
            print("No metric specified")

    def inferno_build_optimizer(self):
        optimizer_dict = self.get('trainer/optimizer')
        for o in optimizer_dict:
            self._trainer.build_optimizer(o, **optimizer_dict[o])

    _INTERVAL_KEYS = ['validate_every', 'save_every', 'evaluate_metric_every']

    def inferno_build_intervals(self):
        for key, arguments in self.get('trainer/intervals').items():
            assert key in self._INTERVAL_KEYS, f'Cannot set interval "{key}". Valid intervals: {self._INTERVAL_KEYS}'
            if isinstance(arguments, dict):
                getattr(self._trainer, key)(**arguments)
            else:
                getattr(self._trainer, key)(arguments)

    def inferno_build_tensorboard(self):
        if self.get('trainer/tensorboard') is not None:
            if TensorboardLogger is None:
                print("warning can not use TensorboardLogger")
                return

            tb_args = self.get('trainer/tensorboard')

            # pop arguments specifying config logging
            log_config = tb_args.pop('log_config', True)
            split_keys = tb_args.pop('split_config_keys', True)

            # pop argument specifying anywhere logging for TensorboardMixin # TODO: how to put this in TensorboardMixin?
            log_anywhere_keys = tb_args.pop('log_anywhere', None)

            tb_args['log_directory'] = f"{self.experiment_directory}/Logs"
            print("logging to ", tb_args['log_directory'])
            tb_logger = TensorboardLogger(**tb_args)

            # register Tensorboard logger
            self._trainer.build_logger(tb_logger)
            # and set _logger to so it can be used by the TensorboardMixin
            self._logger = tb_logger.writer
            if log_config:
                self.log_configuration(split_keys)
            # If experiment also uses TensorboardMixin, register it for anywhere logging
            if isinstance(self, TensorboardMixin):
                register_logger(self, log_anywhere_keys)
            else:
                assert log_anywhere_keys is None, f'Cannot register anywhere logging for keys {log_anywhere_keys}, ' \
                                                  f'please inherit from TensorboardMixin.'

    def log_configuration(self, split_keys=True):
        for filename in os.listdir(self.configuration_directory):
            print(f'logging {filename}')
            if filename.endswith('.yml'):
                with open(os.path.join(self.configuration_directory, filename)) as f:
                    if split_keys:
                        tags = []
                        paragraphs = []
                        for i, line in enumerate(f.readlines()):
                            # add tab to each line to make sure the paragraph is formatted as code
                            if not line.startswith((' ', '#', '\t')) and len(line.split(':')) > 1:
                                paragraphs.append('\t' + ':'.join(line.split(':')[1:]))
                                tags.append(line.split(':')[0])
                            else:
                                paragraphs[-1] += '\t' + line
                        for tag, paragraph in zip(tags, paragraphs):
                            self._logger.add_text(tag=self.get_full_tag('/'.join([tag, filename])),
                                                  text_string=paragraph, global_step=0)
                    else:
                        text = '\t' + f.read().replace('\n', '\n\t')
                        self._logger.add_text(tag=self.get_full_tag(filename), text_string=text,
                                              global_step=0)

    def inferno_build_limits(self):
        if self.get(f'trainer/max_epochs') is not None:
            self._trainer.set_max_num_epochs(self.get(f'trainer/max_epochs'))
        elif self.get(f'trainer/max_iterations') is not None:
            self._trainer.set_max_num_iterations(self.get(f'trainer/max_iterations'))
        else:
            print("No termination point specified!")

    def inferno_build_callbacks(self):
        # build all callbacks from nested conf file
        if self.get('trainer/callbacks') is not None:
            for cb_class in self.get('trainer/callbacks'):
                cb_class_module = locate(cb_class, [inferno.trainers.callbacks])
                for cb in self.get(f'trainer/callbacks/{cb_class}'):
                    print(f'creating trainer/callbacks/{cb_class}/{cb}')
                    args = self.get(f'trainer/callbacks/{cb_class}/{cb}')
                    if "noargs" in args:
                        callback = getattr(cb_class_module, cb)()
                    else:
                        callback = getattr(cb_class_module, cb)(**args)
                    self._trainer.register_callback(callback)

        if self.get('firelight') is not None:
            if firelight_visualizer is None:
                raise ImportError("firelight could not be imported but is present in the config file")
            else:
                # if requested, register anywhere logger for firelight
                print(f"registering firelight for anywhere logging with keys "
                      f"{self.get('firelight').get('log_anywhere', 'all')}")
                register_logger(FirelightLogger(self.trainer), self.get('firelight').pop('log_anywhere', 'all'))
                flc = firelight_visualizer(self.get('firelight'))
                self._trainer.register_callback(flc)

    @property
    def train_loader(self):
        if not hasattr(self, '_train_loader'):
            self._train_loader = self.build_train_loader()
        return self._train_loader

    @property
    def val_loader(self):
        if not hasattr(self, '_val_loader'):
            self._val_loader = self.build_val_loader()
        return self._val_loader

    @property
    def num_targets(self):
        return self.get('trainer/num_targets', 1)

    def inferno_build_loaders(self):
        self._trainer.bind_loader('train',
                                  self.train_loader,
                                  num_targets=self.num_targets)

        if self.val_loader is not None:
            self._trainer.bind_loader('validate',
                                      self.val_loader,
                                      num_targets=self.num_targets)

    def create_transform(self, list_of_transforms):
        return Compose(*[create_instance(t, self.TRANSFORM_LOCATIONS) for t in list_of_transforms])

    def train(self):
        return self.trainer.fit()

import numpy as np


class SimpleInferenceMixin(InfernoMixin):
    def infer_simple(self):


        self.trainer.eval_mode()
        loader_name= "validate"

        # Everytime we should loop over the full dataset (restart generators):
        # TODO: this will probably load a random batch. You maybe want to turn random=False
        loader_iter = self.val_loader.__iter__()

        from inferno.utils import python_utils as pyu

        # Record the epoch we're validating in
        iteration_num = 0
        nb_tot_predictions = len(loader_iter)
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                self.trainer.console.info("Generator exhausted, breaking.")
                break

            # Delay SIGINTs till after computation
            with pyu.delayed_keyboard_interrupt(), torch.no_grad():
                batch = self.trainer.wrap_batch(batch, from_loader=loader_name, volatile=True)
                # Separate
                inputs, target = self.trainer.split_batch(batch, from_loader=loader_name)

                # # Wrap
                # inputs = self.trainer.to_device(inputs)
                outputs = self.trainer.apply_model(*inputs)

                self.trainer.console.progress(
                    "{} of {} inference predictions done".format(iteration_num, nb_tot_predictions))

            # TODO: save the outputs somewhere on disk
            pass

            # TODO: decide if you want to predict more or break
            break




class AffinityInferenceMixin(ParsingMixin):
    def __init__(self, *super_args, **super_kwargs):
        super(AffinityInferenceMixin, self).__init__(*super_args, **super_kwargs)
        self._blending = None

    def infer(self):
        self.trainer.eval_mode()

        # Update loader depending on the output shape:
        self.deduce_crops_and_strides_from_output_size()

        # Everytime we should loop over the full dataset (restart generators):
        loader_iter = self.infer_loader.__iter__()
        full_volume_shape = self.get("full_volume_infer_shape")
        assert isinstance(full_volume_shape, tuple)
        # TODO: (add augmentation)

        # Output files
        full_output = None
        mask = None

        # Record the epoch we're validating in
        iteration_num = 0
        nb_tot_predictions = len(loader_iter)
        while True:
            try:
                inputs, indices = next(loader_iter)
            except StopIteration:
                self.trainer.console.info("Generator exhausted, breaking.")
                break

            assert all([isinstance(indx, IndexSpec) for indx in indices]), "During inference, the dataloader" \
                                                                           "should return the indices"
            self.trainer.console.progress("{} of {} inference predictions done".format(iteration_num, nb_tot_predictions))

            pred, patch_mask = self.get_prediction(inputs)

            for b in range(pred.shape[0]):
                # TODO: not needed, gather does not anyway return anything else but a list of tensors
                if not isinstance(pred, np.ndarray):
                    pred_batch = pred[b].data.cpu().numpy()
                else:
                    pred_batch = pred[b]
                patch_mask_batch = None if patch_mask is None else patch_mask[b].data.cpu().numpy()

                full_output, mask = self.blend_new_patch(full_output, mask,
                                                        pred_batch,
                                                        indices[b].base_sequence_at_index,
                                                         patch_mask=patch_mask_batch)

            iteration_num += 1



        # Crop full output:
        if self.get("inference/global_padding", False):
            # FIXME: assert and place all this stuff in a better place...!!
            ds_ratio = self.get("inference/global_ds_ratio")
            global_padding = self.get("inference/global_padding")
            out_dws_fct = self.get("inference/output_dws_fact", [1, 1, 1])
            global_padding = [[int(pad[0]/dws),int(pad[1]/dws)] for pad, dws in zip(global_padding, out_dws_fct)]
            full_shape = self.get("full_volume_infer_shape")

            crop = (slice(None),) +  tuple(slice(pad[0],
                               full_shape[i] - pad[1],
                               ds_ratio[i]) for i, pad in enumerate(
                global_padding))
            full_output = full_output[crop]
            mask = mask[crop]

        # divide by the mask to normalize all pixels
        assert mask.shape == full_output.shape
        if not (mask != 0).all():
            print("Warning! Not all parts of the dataset have been predicted!")
            # assert (mask != 0).all(), "Not all parts of the dataset have been predicted! " \
            #                       "(Is the stride set correctly..?)"
            full_output[mask == 0] = -1.
            mask[mask == 0] = 1.

        full_output /= mask

        self.save_infer_output(full_output)


    def deduce_crops_and_strides_from_output_size(self):
        # Get the first batch and compare input/output sizes:
        inputs, _ = self.infer_loader.dataset[0]

        # Get original input shape:
        in_shape = self.get("loaders/infer/volume_config/window_size")

        # FIXME: this crap
        if isinstance(inputs, tuple):
            inputs = tuple(inp.unsqueeze(0) for inp in inputs)
        elif isinstance(inputs, list):
            inputs = list(inp.unsqueeze(0) for inp in inputs)
        else:
            inputs = inputs.unsqueeze(0)
        outputs, _ = self.get_prediction(inputs)

        # Process outputs:
        # if isinstance(inputs, (tuple, list)):
        #     print("WARNING: considering the first input!")
        #     inputs = inputs[0]
        if isinstance(outputs, (tuple, list)):
            print("WARNING: considering the first output!")
            outputs = outputs[0]
        # in_shape = inputs.shape[-3:]
        out_shape = outputs.shape[-3:]

        # Get local crop config (can be asymmetric!)
        local_crop = self.get("inference/crop_prediction")
        if hasattr(self.model, "final_asym_crop_pred"):
            model_local_crop = self.model.final_asym_crop_pred
            if local_crop is None:
                local_crop = model_local_crop
            else:
                # Combine two crops:
                local_crop = [[crp[0]+mdl_crp[0], crp[1]+mdl_crp[1]] for crp, mdl_crp in zip(local_crop, model_local_crop)]


        if local_crop is not None:
            # slicing w.r.t the current output
            assert isinstance(local_crop, (tuple, list))
            assert len(local_crop) == 3 and all([len(cr) == 2 for cr in local_crop])
            local_slicing = tuple(slice(pad[0],
                                        out_shape[i] - pad[1])
                                  for i, pad in enumerate(local_crop))
            self.set("inference/crop_prediction", local_crop)
            out_shape_cropped = outputs[(slice(None), slice(None)) + local_slicing].shape[-3:]
        else:
            out_shape_cropped = outputs.shape[-3:]
            local_crop = [[0,0], [0,0], [0,0]]

        out_dws_fct = self.get("inference/output_dws_fact", [1,1,1])
        out_shape_scaled = tuple(int(sh*dws) for sh, dws in zip(out_shape_cropped, out_dws_fct))

        # Update the stride:
        window_overlap = self.get("inference/window_overlap", [0, 0, 0])
        assert len(window_overlap) == 3
        stride = [shp-ovrlap*2 for shp, ovrlap in zip(list(out_shape_scaled), window_overlap)]
        assert all(strd>0 for strd in stride), "Too much overlap for the current output shape of the model: " \
                                               "{}, {}".format(window_overlap, out_shape_scaled)
        self.set("loaders/infer/volume_config/stride", stride)
        print("Shape of final prediction: {}. In the original res: {}".format(out_shape_cropped, out_shape_scaled))
        print("Stride: {}.".format(stride))

        # Deduce global crop, that should be in the output resolution:
        input_shape_in_out_res = tuple(int(sh/dws_out) for sh, dws_out in zip(in_shape, out_dws_fct))
        global_crop = deepcopy(local_crop)
        diff = [in_sh-out_sh-pad[0]-pad[1] for in_sh, out_sh, pad in zip(input_shape_in_out_res, out_shape_cropped, global_crop)]
        assert all(d % 2 == 0 for d in diff), "Something unexpected happened while cropping"
        global_crop = [[int(d/2)+pad[0], int(d/2)+pad[1]] for d, pad in zip(diff, global_crop)]

        # Save deduced crop for later:
        self.set("inference/crop_global_slice", global_crop)
        
        # Re-build the inference-loader after setting the new stride:
        self._infer_loader = self.build_infer_loader()



    def get_prediction(self, inputs):
        # Delay SIGINTs till after computation
        with pyu.delayed_keyboard_interrupt(), torch.no_grad():
            # Wrap
            inputs = self.trainer.to_device(inputs)
            if isinstance(inputs, (tuple, list)):
                patch_outputs = self.trainer.apply_model(*inputs)
            else:
                patch_outputs = self.trainer.apply_model(inputs)

        # FIXME: improve this
        if self.get("inference/return_patch_mask"):
            assert isinstance(patch_outputs, tuple)
            assert len(patch_outputs) == 2
            pred, mask = patch_outputs
        else:
            mask = None
            if isinstance(patch_outputs, list):
                pred = patch_outputs[0]
            else:
                pred = patch_outputs
        pred = self.postprocess_patch_outputs(pred)

        return pred, mask

    def save_infer_output(self, output):
        pass

    def postprocess_patch_outputs(self, patch_outputs):
        return patch_outputs

    def blend_new_patch(self, full_output, mask, patch_output, slicing, patch_mask=None):
        assert isinstance(patch_output, np.ndarray)
        assert patch_output.ndim == 4

        nb_channels = patch_output.shape[0]
        patch_shape = patch_output.shape[1:]
        if full_output is None:
            full_output = np.zeros((nb_channels,) + self.get("full_volume_infer_shape"), dtype='float32')
        if mask is None:
            mask = np.zeros((nb_channels,) + self.get("full_volume_infer_shape"), dtype='float32')

        local_slicing, global_slicing = self.get_slicings(slicing, patch_shape)

        # Apply local slicing in any case:
        patch_output = patch_output[local_slicing]

        if patch_mask is not None:
            assert self.blending is None, "Outputed masks and blending are not supported at the same time"
            mask[global_slicing] += patch_mask[local_slicing]
            patch_output *= patch_mask[local_slicing]
        else:
            if self.blending is not None:
                patch_output, blending_mask = self.blending(patch_output)
                mask[global_slicing] += blending_mask
            else:
                mask[global_slicing] += 1
        # add up predictions in the output
        full_output[global_slicing] += patch_output

        return full_output, mask

    @property
    def blending(self):
        if self._blending is not None:
            return self._blending
        else:
            if self.get("inference/blending_kwargs", None) is not None:
                # TODO: fix reference...
                from neurofire.inference.blending import Blending
                blending_kwargs = self.get("inference/blending_kwargs")
                if "ramp_size" not in blending_kwargs:
                    window_overlap = self.get("inference/window_overlap", None)
                    assert window_overlap is not None, "No ramp size could be deduced for blending"
                    blending_kwargs['ramp_size'] = window_overlap
                self._blending = Blending(**blending_kwargs)
                return self._blending
                # TODO: init _blending
            else:
                return None

    def get_slicings(self, slicing, shape):
        slice_crop = self.get("inference/crop_global_slice", False)
        prediction_crop = self.get("inference/crop_prediction", False)
        assert all([slicing[i].step == 1 for i in range(3)]), "Downscaling option not implemented yet!"

        # Check whether the output has a different resolution, then rescale global slicing:
        out_dws_fct = self.get("inference/output_dws_fact", [1,1,1])
        # print(slicing)
        # assert all(all((slc.start % dws == 0, slc.stop % dws == 0)) for slc, dws in zip(slicing, out_dws_fct))
        global_slicing = tuple(slice(int(slc.start / dws), int(slc.stop / dws) ) for slc, dws in zip(slicing, out_dws_fct))

        # crop away the padding (we treat global as local padding) if specified
        # this is generally not necessary if we use blending
        if slice_crop:
            # slicing w.r.t the global output
            assert isinstance(slice_crop, (tuple, list))
            assert len(slice_crop) == 3 and all([len(cr) == 2 for cr in slice_crop])
            global_slicing = tuple(slice(global_slicing[i].start + pad[0],
                                         global_slicing[i].stop - pad[1])
                                   for i, pad in enumerate(slice_crop))
        if prediction_crop:
            # slicing w.r.t the current output
            assert isinstance(prediction_crop, (tuple, list))
            assert len(prediction_crop) == 3 and all([len(cr) == 2 for cr in prediction_crop])
            local_slicing = tuple(slice(pad[0],
                                        shape[i] - pad[1])
                                  for i, pad in enumerate(prediction_crop))
        else:
            local_slicing = tuple(slice(None, None) for _ in range(3))

        # Add channel dimension:
        local_slicing = (slice(None), ) + local_slicing
        global_slicing = (slice(None), ) + global_slicing

        return local_slicing, global_slicing

    @property
    def trainer(self):
        """
        inferno trainer. Will be loaded from file on first use.
        """
        if inferno is None:
            raise ModuleNotFoundError("InfernoMixin requires inferno. You can "
                                      "install it with `pip install in "
                                      "pip install inferno-pytorch`")
        # Build trainer if it doesn't exist
        if not hasattr(self, '_trainer'):

            # noinspection PyAttributeOutsideInit
            self._trainer = inferno.trainers.basic.Trainer(self.model)
            # self._trainer = inferno.trainers.basic.Trainer() \
            #     .load(self.get("inference/path_checkpoint_trainer"), best=True)

            self._trainer.to(self.device)

        return self._trainer

    def build_loader(self):
        trainer = self.trainer
        trainer.bind_loader('test',
                                  self.infer_loader)

        if self.val_loader is not None:
            self._trainer.bind_loader('validate',
                                      self.val_loader,
                                      num_targets=self.num_targets)


    @property
    def device(self):
        # TODO: move to ParsingMixin?
        if self._device is None:
            # noinspection PyAttributeOutsideInit
            self._device = torch.device(self.get('device'))
        return self._device

    @property
    def infer_loader(self):
        if not hasattr(self, '_infer_loader'):
            self._infer_loader = self.build_infer_loader()
            out_dws_fact = self.get("inference/output_dws_fact", [1,1,1])
            self.set("full_volume_infer_shape", tuple(int(sh/scl) for sh, scl in zip(self._infer_loader.dataset.volume.shape, out_dws_fact)))
            self.set("inference/global_padding", self._infer_loader.dataset.padding)
            self.set("inference/global_ds_ratio", self._infer_loader.dataset.downsampling_ratio)

        return self._infer_loader
