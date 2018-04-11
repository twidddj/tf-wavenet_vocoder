"""
the code is adapted from:
https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py (base model)
https://github.com/twidddj/vqvae/blob/master/wavenet/model.py
"""

import numpy as np
import tensorflow as tf

from .ops import causal_conv, mu_law_encode
from .mixture import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

PREFIX_QUEUE_VAR = "Q_L"

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer()
    variable = tf.get_variable(name, shape, initializer=initializer)
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.get_variable(name, shape, initializer=initializer)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        variable = tf.get_variable(name, initializer=initial_val)
        return variable
    else:
        return create_variable(name, shape)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 out_channels=None,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=None,
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None,
                 local_condition_channels=80):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                initial convolution.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        assert filter_width > 1

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.out_channels = out_channels or quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width or filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.local_condition_channels = local_condition_channels

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)

        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            if self.global_condition_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_condition_cardinality,
                         self.global_condition_channels])
                    var['embeddings'] = layer

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                else:
                    initial_channels = self.quantization_channels
                layer['filter'] = create_variable(
                    'filter',
                    [self.initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        
                        current['cond_filter'] = create_variable('cond_filter', [1, self.local_condition_channels, self.dilation_channels])
                        current['cond_gate'] = create_variable('cond_gate', [1, self.local_condition_channels, self.dilation_channels])
                        if self.use_biases:
                            current['cond_filter_bias'] = create_bias_variable(
                                'cond_filter_bias',
                                [self.dilation_channels])
                            current['cond_gate_bias'] = create_bias_variable(
                                'cond_gate_bias',
                                [self.dilation_channels])

                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.out_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.out_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation, local_condition_batch,
                               global_condition_batch, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)
        
        if local_condition_batch is not None:
            condition_cut = tf.shape(local_condition_batch)[1] - tf.shape(conv_filter)[1]
            lc = tf.slice(local_condition_batch, [0, condition_cut, 0], [-1, -1, -1])
            conv_filter += tf.nn.conv1d(lc, variables['cond_filter'], stride=1, padding="SAME", name="cond_filter")
            conv_gate += tf.nn.conv1d(lc, variables['cond_gate'], stride=1, padding="SAME", name="cond_gate")
            if self.use_biases:
                conv_filter += variables['cond_filter_bias']
                conv_gate += variables['cond_gate_bias']

        if global_condition_batch is not None:
            conv_filter += tf.nn.conv1d(global_condition_batch, variables['gc_filtweights'], stride=1, padding="SAME",
                                        name="gc_filter")
            conv_gate += tf.nn.conv1d(global_condition_batch, variables['gc_gateweights'], stride=1, padding="SAME",
                                      name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter += filter_bias
            conv_gate += gate_bias

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)
                tf.histogram_summary(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights, is_initial=False):
        '''Perform convolution for a single convolutional processing step.'''

        if state_batch is not None:
            output = tf.matmul(state_batch[0], weights[0])
            filter_width = self.initial_filter_width if is_initial else self.filter_width

            i = 0  # This value will be used when filter width == 2
            for i in range(1, filter_width - 1):
                output += tf.matmul(state_batch[i], weights[i])
            i = i+1
        else:
            output = 0
            i = 0

        output += tf.matmul(input_batch, weights[i])

        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter, is_initial=True)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  local_condition_batch, global_condition_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(input_batch, state_batch, weights_gate)
        
        if local_condition_batch is not None:
            output_filter += tf.matmul(local_condition_batch, variables['cond_filter'][0, :, :])
            output_gate += tf.matmul(local_condition_batch, variables['cond_gate'][0, :, :])
            if self.use_biases:
                output_filter += variables['cond_filter_bias']
                output_gate += variables['cond_gate_bias']

        if global_condition_batch is not None:
            output_filter += tf.matmul(global_condition_batch, variables['gc_filtweights'][0, :, :])
            output_gate += tf.matmul(global_condition_batch, variables['gc_gateweights'][0, :, :])

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :], name="TEST_transform")
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :], name="TEST_skip")
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def create_network(self, input_batch, local_condition_batch, global_condition_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation, local_condition_batch,
                        global_condition_batch, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('postprocess1_weights', w1)
                tf.histogram_summary('postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('postprocess1_biases', b1)
                    tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def _create_queue(self, dilation, n_channel, batch_size, is_initial=False, name=None):
        filter_width = self.initial_filter_width if is_initial else self.filter_width

        if filter_width == 1:
            return None

        shape = (dilation * (filter_width - 1), batch_size, n_channel)
        value = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial_value=value, name=name, trainable=False)

    def _create_q_ops(self, batch_size):
        qs = []
        input_channels = 1 if self.scalar_input else self.quantization_channels
        q = self._create_queue(1, input_channels, batch_size,
                               is_initial=True,
                               name=PREFIX_QUEUE_VAR + str(0))

        qs.append(q)

        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    q = self._create_queue(dilation, self.residual_channels, batch_size,
                                           name=PREFIX_QUEUE_VAR + str(layer_index + 1))
                    qs.append(q)

        return qs

    def _update(self, q, current_q_idx, x, is_initial=False):
        if q is None:
            return None

        filter_width = self.initial_filter_width if is_initial else self.filter_width

        # dequeue
        for i in range(1, filter_width - 1):
            idx = current_q_idx + i
            q = tf.scatter_update(q, idx - 1, q[idx])

        # enqueue
        q = tf.scatter_update(q, current_q_idx + (filter_width - 2), x)

        return q

    def create_update_q_ops(self, qs, initial, others, gen_num, batch_size=1):
        current_q_idx = 0

        # Initial queue value will be None if initial filter width == 1
        if self.initial_filter_width > 1:
            q = qs[0]
            input_channels = 1 if self.scalar_input else self.quantization_channels
            ipt = tf.reshape(initial, [batch_size, input_channels])
            q = self._update(q, current_q_idx, ipt, is_initial=True)
            qs[0] = q

        for layer_index, dilation in enumerate(self.dilations):
            q = qs[layer_index + 1]
            current_q_idx = (gen_num % dilation) * (self.filter_width - 1)
            ipt = tf.reshape(others[layer_index], [batch_size, self.residual_channels])
            q = self._update(q, current_q_idx, ipt)
            qs[layer_index + 1] = q

        if self.initial_filter_width == 1:
            return qs[1:]
        else:
            return qs

    @staticmethod
    def get_vars_q():
        return list(filter(lambda var: var.name.split('/')[-1].startswith(PREFIX_QUEUE_VAR),
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def _create_generator(self, qs, x, gen_num, c, g):
        outputs = []
        output_layers = []

        q = qs[0]

        current_q_idx = 0
        current_data_idx = current_q_idx + (self.initial_filter_width - 1)

        current_layer = x

        if q is not None:
            current_state = q[current_q_idx:current_data_idx]
        else:
            current_state = None

        output_layers.append(current_layer)

        current_layer = self._generator_causal_layer(current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    q = qs[layer_index + 1]
                    current_q_idx = (gen_num % dilation) * (self.filter_width - 1)
                    current_data_idx = current_q_idx + (self.filter_width - 1)
                    output_layers.append(current_layer)

                    current_state = q[current_q_idx:current_data_idx]
                    #                 current_layer = tf.Print(current_layer, [current_layer],
                    #                                          message="current_layer{}:".format(layer_index + 1))

                    output, current_layer = self._generator_dilation_layer(current_layer, current_state, layer_index, c, g)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2

        return conv2, output_layers

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding,
                [self.batch_size, 1, self.global_condition_channels])

        return embedding

    def predict_proba(self, waveform, local_condition=None, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.reshape(waveform, [self.batch_size, -1, 1])
                encoded = tf.cast(encoded, tf.float32)
            else:
                encoded = self._one_hot(waveform)

            gc_embedding = self._embed_gc(global_condition)
            raw_output = self.create_network(encoded, local_condition, gc_embedding)

            if self.scalar_input:
                out = tf.reshape(raw_output, [self.batch_size, -1, self.out_channels])
                last = sample_from_discretized_mix_logistic(out)
            else:
                out = tf.reshape(raw_output, [-1, self.out_channels])
                # Cast to float64 to avoid bug in TensorFlow
                proba = tf.cast(
                    tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
                last = tf.slice(
                    proba,
                    [tf.shape(proba)[0] - 1, 0],
                    [1, self.out_channels])
            return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, gen_num, batch_size=1,
                                  local_condition=None, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''

        q_ops = self._create_q_ops(batch_size)

        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            else:
                encoded = self._one_hot(waveform)
                encoded = tf.reshape(encoded, [-1, self.quantization_channels])

            gc_embedding = self._embed_gc(global_condition)
            if gc_embedding is not None:
                gc_embedding = tf.squeeze(gc_embedding, [1])
            raw_output, output_layers = self._create_generator(q_ops, encoded, gen_num, local_condition, gc_embedding)

            if self.scalar_input:
                out = tf.reshape(raw_output, [batch_size, -1, self.out_channels])
                proba = sample_from_discretized_mix_logistic(out)
            else:
                out = tf.reshape(raw_output, [-1, self.out_channels])
                proba = tf.cast(tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            return proba, output_layers, q_ops

    def loss(self,
             input_batch,
             local_condition_batch=None,
             global_condition_batch=None,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            gc_embedding = self._embed_gc(global_condition_batch)

            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                encoded_input = mu_law_encode(input_batch, self.quantization_channels)
                encoded = self._one_hot(encoded_input)
                network_input = encoded

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            inputs = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            raw_output = self.create_network(inputs, local_condition_batch, gc_embedding)

            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                target_output = tf.slice(
                    network_input,
                    [0, self.receptive_field, 0],
                    [-1, -1, -1])

                if self.scalar_input:
                    loss = discretized_mix_logistic_loss(raw_output, target_output,
                                                         num_class=2**16, reduce=False)
                    reduced_loss = tf.reduce_mean(loss)
                else:
                    target_output = tf.reshape(target_output, [-1, self.out_channels])
                    prediction = tf.reshape(raw_output,  [-1, self.out_channels])
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss
