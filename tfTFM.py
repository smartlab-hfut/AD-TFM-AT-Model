import tensorflow as tf
import numpy as np
import math


class AdaTFM():

    def __init__(self, state_size, input_size, dim_size, target_size,omega, K, J, dtype=tf.float64):
        '''
        Parameters
        -------
        state_size: int (unit_size D)
            - size of timesteps(input_size)
        input_size: int
            - size of input features(dim_size)
        target_size: int
            - size of the output space
        dtype: tf data type
            - default tf.float64, usually follows from input data type
        '''
        self.state_size = state_size
        self.input_size = input_size
        self.dim_size = dim_size
        self.target_size = target_size
        self.omega = omega
        self.K = K
        self.J = J
        self.dtype = dtype
        self.build_graph()

    def build_graph(self):
        """
        Build TensorFlow graph and expose operations as class methods

        Returns
        -------
        - Set
            * self.loss = loss_func
            * self.train_op = train_op
            * self.accuracy = accuracy
        """
        # -----Sequence Input------
        self._inputs = tf.placeholder(self.dtype, shape=[None,
                                                         self.input_size,
                                                         self.dim_size])
        self.ys = tf.placeholder(self.dtype, shape=[None,
                                                    self.target_size])
        # ---------State-----------
        self.W_state = tf.Variable(tf.zeros([self.dim_size,
                                             self.state_size], dtype=self.dtype))
        self.V_state = tf.Variable(tf.zeros([self.state_size,
                                             self.state_size], dtype=self.dtype))
        self.b_state = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # --------Frequency--------
        self.W_freq = tf.Variable(tf.zeros([self.dim_size,
                                            self.K], dtype=self.dtype))
        self.V_freq = tf.Variable(tf.zeros([self.state_size,
                                            self.K], dtype=self.dtype))
        self.b_freq = tf.Variable(tf.ones([self.K], dtype=self.dtype))
        # --------Class--------
        self.W_class = tf.Variable(tf.zeros([self.dim_size,
                                            self.J], dtype=self.dtype))
        self.V_class = tf.Variable(tf.zeros([self.state_size,
                                            self.J], dtype=self.dtype))
        self.b_class = tf.Variable(tf.ones([self.J], dtype=self.dtype))
        # -------Modulation-------
        self.W_g = tf.Variable(tf.zeros([self.dim_size,
                                         self.state_size], dtype=self.dtype))
        self.V_g = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_g = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Input-----------
        self.W_i = tf.Variable(tf.zeros([self.dim_size,
                                         self.state_size], dtype=self.dtype))
        self.V_i = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_i = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))

        # ---------Output----------
        self.W_c = tf.Variable(tf.zeros([self.K,
                                         self.J,
                                         self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_c = tf.Variable(tf.ones([self.K,
                                        self.J,
                                        self.state_size], dtype=self.dtype))
        self.W_o = tf.Variable(tf.zeros([self.dim_size,
                                         self.state_size], dtype=self.dtype))
        self.V_o = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_o = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # -------Step Output-------
        self.W_z = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_z = tf.Variable(tf.ones([self.state_size,
                                        self.state_size], dtype=self.dtype))

        # ---------State_1-----------
        self.W_state_1 = tf.Variable(tf.zeros([self.state_size,
                                              self.state_size], dtype=self.dtype))
        self.V_state_1 = tf.Variable(tf.zeros([self.state_size,
                                              self.state_size], dtype=self.dtype))
        self.b_state_1 = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # --------Frequency_1--------
        self.W_freq_1 = tf.Variable(tf.zeros([self.state_size,
                                             self.K], dtype=self.dtype))
        self.V_freq_1 = tf.Variable(tf.zeros([self.state_size,
                                             self.K], dtype=self.dtype))
        self.b_freq_1 = tf.Variable(tf.ones([self.K], dtype=self.dtype))
        # --------Class_1--------
        self.W_class_1 = tf.Variable(tf.zeros([self.state_size,
                                              self.J], dtype=self.dtype))
        self.V_class_1 = tf.Variable(tf.zeros([self.state_size,
                                              self.J], dtype=self.dtype))
        self.b_class_1 = tf.Variable(tf.ones([self.J], dtype=self.dtype))
        # -------Modulation_1-------
        self.W_g_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.V_g_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.b_g_1= tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Input_1-----------
        self.W_i_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.V_i_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.b_i_1 = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))

        # ---------Output_1----------
        self.W_c_1 = tf.Variable(tf.zeros([self.K,
                                          self.J,
                                          self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.b_c_1 = tf.Variable(tf.ones([self.K,
                                         self.J,
                                         self.state_size], dtype=self.dtype))
        self.W_o_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.V_o_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.b_o_1 = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # -------Step Output_1-------
        self.W_z_1 = tf.Variable(tf.zeros([self.state_size,
                                          self.state_size,
                                          self.state_size], dtype=self.dtype))
        self.b_z_1 = tf.Variable(tf.ones([self.state_size,
                                          self.state_size], dtype=self.dtype))
        # ------Sequence Output----
        self.W_z_z = tf.Variable(tf.truncated_normal([self.state_size,
                                                      self.target_size],
                                                     dtype=self.dtype,
                                                     mean=0, stddev=.01))
        self.b_z_z = tf.Variable(tf.truncated_normal([self.target_size],
                                                     mean=1, stddev=.01,
                                                     dtype=self.dtype))
        #-------omega--------
        self.W_a = tf.Variable(tf.zeros([self.K,
                                             self.J], dtype=self.dtype))
        self.b_a = tf.Variable(tf.ones([self.K], dtype=self.dtype))

        self.W_b = tf.Variable(tf.zeros([self.K,
                                         self.J], dtype=self.dtype))
        self.b_b = tf.Variable(tf.ones([self.K], dtype=self.dtype))

        '''
        Init hidden state from input dimensions
            - tf.stack() only accepts input of same dimension,
            - Need to make 4 states of dimensions
            - Final dim is (4, samples, timesteps, timesteps)
        '''
        self.air = tf.reduce_sum(self._inputs, [1, 2])
        self.air = tf.stack([self.air for _ in range(self.K)], axis=1)
        self.air = tf.stack([self.air for _ in range(self.J)], axis=2)

        self.air_ = tf.reduce_sum(self._inputs, [1, 2])
        self.air_ = tf.stack([self.air_ for _ in range(self.J)], axis=0)
        self.air_ = tf.stack([self.air_ for _ in range(self.state_size)], axis=2)

        self.init_hidden_t = tf.reduce_sum(self._inputs, [1, 2])
        self.init_hidden_t = tf.stack([self.init_hidden_t for _ in range(self.state_size)], axis=1)
        self.init_hidden_t = tf.stack([self.init_hidden_t for _ in range(self.K)], axis=2)
        self.init_hidden_t = tf.stack([self.init_hidden_t for _ in range(self.J)], axis=3)
        self.init_hidden_t = tf.zeros(tf.shape(self.init_hidden_t), dtype=self.dtype)
        self.init_hidden = tf.stack([self.init_hidden_t,
                                     self.init_hidden_t,
                                     self.init_hidden_t], axis=0)




        self._inp = tf.transpose(tf.transpose(self._inputs, perm=[2, 0, 1]))
        step_freq = tf.cast((np.arange(self.input_size, dtype=np.float64) + 1)
                            / self.input_size, dtype=self.dtype)


        out = tf.scan(self._step,
                      elems=[self._inp, step_freq],
                      initializer=self.init_hidden)

        out = tf.squeeze(out[:, -1:, :, :, -1:, -1:], axis=[1, 4, 5])
        '''
        out_1 = tf.scan(self._step_1,
                        elems=[out, step_freq],
                        initializer=self.init_hidden)

        out_1 = tf.squeeze(out_1[:, -1:, :, :, -1:, -1:], axis=[1, 4, 5])
        # print(out_1)        
        
        '''
        map_fn = lambda _state: tf.nn.relu(tf.matmul(_state, self.W_z_z)
                                           + self.b_z_z)
        all_out = tf.map_fn(map_fn, out)

        attention_size = 64

        def attention(inputs, attention_size, time_major=False):
            if isinstance(inputs, tuple):
                inputs = tf.concat(inputs, 2)
            if time_major:  # (T,B,D) => (B,T,D)
                inputs = tf.transpose(inputs, [1, 0, 2])
            hidden_size = inputs.shape[2].value
            # Trainable parameters
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], dtype=tf.float64, stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attention_size], dtype=tf.float64, stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attention_size], dtype=tf.float64, stddev=0.1))
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
            # the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

            return output, alphas

        self.final_output, llx = attention(all_out, attention_size, time_major=True)
            # transform to batch_size * sequence_length
            # print(self.final_output)

        at_out = self.final_output
        #print(all_out)

        #print(last_out)

        softmax = tf.nn.softmax(at_out)
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys, logits=softmax))
        cross_entropy = -tf.reduce_mean(self.ys * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)))
        train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.ys, 1),
                                      tf.argmax(softmax, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        model_pred = tf.argmax(softmax, 1)
        model_True = tf.argmax(self.ys, 1)
        self.y_score = softmax
        self.model_pred = model_pred
        self.model_True = model_True

        self.loss = cross_entropy
        self.train_op = train_op
        self.accuracy = accuracy

    def _step(self, prev, input):
        '''
        One time step along the sequence time axis

        Parameters
        -------
        prev: tf.stack
        input: tf.stack

        Returns
        -------
        out: tf.stack
            - Need to stack output vector to match size of state for output
        '''
        x_, t_ = input
        real_, img_, z_ = tf.unstack(prev)
        z_ = z_[:, :, 1, 1]
        state_fg = tf.sigmoid(tf.matmul(x_, self.W_state)
                              + tf.matmul(z_, self.V_state)
                              + self.b_state)

        freq_fg = tf.sigmoid(tf.matmul(x_, self.W_freq)
                                + tf.matmul(z_, self.V_freq)
                                + self.b_freq)

        class_fg = tf.sigmoid(tf.matmul(x_, self.W_class)
                                + tf.matmul(z_, self.V_class)
                                + self.b_class)
        fg = self.__outer(state_fg, self._outer(freq_fg, class_fg))
        inp_g = tf.sigmoid(tf.matmul(x_, self.W_g)
                            + tf.matmul(z_, self.V_g)
                            + self.b_g)

        mod_g = tf.tanh(tf.matmul(x_, self.W_i)
                        + tf.matmul(z_, self.V_i)
                        + self.b_i)
        ing = tf.reshape(inp_g * mod_g,[-1,4,4])
        aw = tf.tanh(tf.matmul(ing, self.W_a) + self.b_a)
        aw1 = tf.reshape(aw, [tf.size(aw)])
        print(aw1)
        b1 = (tf.matmul(ing, self.W_b) + self.b_b)
        b2 = b1
        b = tf.reshape(b2, [tf.size(b2)])

        # omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega
        # dwt change to matrix

        cos_wt_, sin_wt_ =  self._wt(self.K, self.J, t_,self.omega,aw1,b)
        cos_wt = self.air * cos_wt_
        sin_wt = self.air * sin_wt_

        real = fg * real_ + self.__outer(inp_g * mod_g, cos_wt)
        img = fg * img_ + self.__outer(inp_g * mod_g, sin_wt)

        amp = tf.sqrt(tf.add(tf.square(real), tf.square(img)))
        amp = tf.transpose(amp, perm=[2, 3, 0, 1])

        def __step(c_k, inputs):
                W_k, b_k, A_k = inputs
                c = tf.tanh(tf.matmul(A_k, W_k) + b_k)
                cc = c_k + c
                return tf.stack(cc)

        def ___step(prev, inputs):

            W_j, b_j, A_j = inputs
            c_j = tf.scan(__step,
                          elems=[W_j, b_j, A_j],
                          initializer=tf.zeros(tf.shape(z_), dtype=self.dtype))
            return c_j

        c = tf.scan(___step,
                    elems=[self.W_c, self.b_c, amp],
                    initializer=tf.zeros(tf.shape(self.air_), dtype=self.dtype))

        last_c = c[-1, -1, :, :]
        output_gate = tf.sigmoid(tf.matmul(x_, self.W_o)
                                 + tf.matmul(z_, self.V_o)
                                 + self.b_o)
        last_z = output_gate * last_c
        # Match dim of state matrix
        last_z = tf.stack([last_z for _ in range(self.K)], axis=2)
        last_z = tf.stack([last_z for _ in range(self.J)], axis=3)
        out = tf.stack([real, img, last_z])

        return out

    def _step_1(self, prev, input):
        '''
        One time step along the sequence time axis

        Parameters
        -------
        prev: tf.stack
        input: tf.stack

        Returns
        -------
        out: tf.stack
            - Need to stack output vector to match size of state for output
        '''
        x_, t_ = input
        real_, img_, z_ = tf.unstack(prev)
        z_ = z_[:, :, 1, 1]
        state_fg = tf.sigmoid(tf.matmul(x_, self.W_state_1)
                              + tf.matmul(z_, self.V_state_1)
                              + self.b_state_1)

        freq_fg = tf.sigmoid(tf.matmul(x_, self.W_freq_1)
                             + tf.matmul(z_, self.V_freq_1)
                             + self.b_freq_1)

        class_fg = tf.sigmoid(tf.matmul(x_, self.W_class_1)
                              + tf.matmul(z_, self.V_class_1)
                              + self.b_class_1)
        fg = self.__outer(state_fg, self._outer(freq_fg, class_fg))
        inp_g = tf.sigmoid(tf.matmul(x_, self.W_g_1)
                           + tf.matmul(z_, self.V_g_1)
                           + self.b_g_1)

        mod_g = tf.tanh(tf.matmul(x_, self.W_i_1)
                        + tf.matmul(z_, self.V_i_1)
                        + self.b_i_1)

        # omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega
        # dwt change to matrix

        cos_wt_, sin_wt_ = self._wt(self.K, self.J, t_)
        cos_wt = self.air * cos_wt_
        sin_wt = self.air * sin_wt_

        # tf.cos(self.omega * t_)  t_gai
        real = fg * real_ + self.__outer(inp_g * mod_g, cos_wt)
        img = fg * img_ + self.__outer(inp_g * mod_g, sin_wt)

        amp = tf.sqrt(tf.add(tf.square(real), tf.square(img)))
        # Transpose to dim (frequency_components, samples, state) for scan
        amp = tf.transpose(amp, perm=[2, 3, 0, 1])
        # Frequency Step Kernel

        def __step(c_k, inputs):

                W_k, b_k, A_k = inputs
                c = tf.tanh(tf.matmul(A_k, W_k) + b_k)
                cc = c_k + c
                return tf.stack(cc)

        def ___step(prev, inputs):

            W_j, b_j, A_j = inputs
            c_j = tf.scan(__step,
                          elems=[W_j, b_j, A_j],
                          initializer=tf.zeros(tf.shape(z_), dtype=self.dtype))
            return c_j

        c = tf.scan(___step,
                    elems=[self.W_c_1, self.b_c_1, amp],
                    initializer=tf.zeros(tf.shape(self.air_), dtype=self.dtype))

        last_c = c[-1, -1, :, :]
        output_gate = tf.sigmoid(tf.matmul(x_, self.W_o_1)
                                 + tf.matmul(z_, self.V_o_1)
                                 + self.b_o_1)
        last_z = output_gate * last_c
        # Match dim of state matrix
        last_z = tf.stack([last_z for _ in range(self.K)], axis=2)
        last_z = tf.stack([last_z for _ in range(self.J)], axis=3)
        out_1 = tf.stack([real, img, last_z])

        return out_1

    def _outer(self, x, y):
        '''
        Outer Product of 2 2D matrix

        Parameters
        -------
        x: tf.tensor
            - 2D Shape
        y: tf.tensor
            - 2D Shape

        Returns
        -------
        out: tf.tensor
            - 3D Shape
        '''
        out = x[:, :, np.newaxis] * y[:, np.newaxis, :]

        return out

    def __outer(self, x, y):
        '''
        Outer Product of 2 2D matrix

        Parameters
        -------
        x: tf.tensor
            - 2D Shape
        y: tf.tensor
            - 3D Shape

        Returns
        -------
        out: tf.tensor
            - 4D Shape
        '''
        x = tf.stack([x for _ in range(self.K)], axis=2)
        x = tf.stack([x for _ in range(self.J)], axis=3)
        y = tf.stack([y for _ in range(self.state_size)], axis=1)
        out = x * y

        return out

    def _wt(self, K, J, t,omega,aw1,b):

        self.omega=omega
        self.b=b
        self.aw=aw1
        i = 0

        a = [tf.Variable(0, dtype=tf.float64) for i in range(K * J)]
        a1 = [tf.Variable(0, dtype=tf.float64) for i in range(K * J)]
        b = [tf.Variable(0, dtype=tf.float64) for i in range(K * J)]
        b1 = [tf.Variable(0, dtype=tf.float64) for i in range(K * J)]
        for k in range(K):
            for j in range(J):
                a1[i] = tf.compat.v1.assign(a[i], tf.cos(self.omega*self.aw[i] * ((t+self.b[i]) / 2 ** j - k)) * tf.exp(
                    -(((t+self.b[i]) / 2 ** j - k) ** 2) / 2))
                b1[i] = tf.compat.v1.assign(b[i], tf.sin(self.omega*self.aw[i] * ((t+self.b[i]) / 2 ** j - k)) * tf.exp(
                    -(((t+self.b[i]) / 2 ** j - k) ** 2) / 2))
                i = i + 1

        a2 = tf.reshape(a1, [K, J])
        b2 = tf.reshape(b1, [K, J])
        return a2, b2




