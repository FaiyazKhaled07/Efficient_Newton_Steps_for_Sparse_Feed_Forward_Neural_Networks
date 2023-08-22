def jacobian(self, x, threshold=1e-5):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            tape.watch(x)
            output = self.call(x)
        jac = tape.batch_jacobian(output, x)
        w = tf.where(tf.abs(self.mat) > threshold, self.mat, tf.zeros_like(self.mat))
        jac_sparse = tf.matmul(w, jac, transpose_a=True)
        jac_sparse = tf.transpose(jac_sparse, perm=[0, 2, 1])
        diag = tf.linalg.diag_part(jac_sparse)
        lower = tf.linalg.band_part(jac_sparse, 1, 0)
        upper = tf.linalg.band_part(jac_sparse, 0, 1)
        tridiag = tf.linalg.set_diag(tf.zeros_like(jac_sparse), diag) + lower + upper
        return tridiag
