import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
#import lsh_matmul_grad

try:
  from tensorflow_lsh_matmul.python.ops.lsh_matmul_op import lsh_matmul
except ImportError:
  from lsh_matmul_op import lsh_matmul

#lsh_matmul_module = tf.load_op_library('./_lsh_matmul_op.so')

class LshMatmulTest(test.TestCase):
    '''
    def test_raisesExceptionWithIncompatibleDimensions(self):
        with self.session():
            with self.assertRaises(ValueError):
                lsh_matmul([1, 2], [[1, 2], [3, 4]]).eval()
            with self.assertRaises(ValueError):
                self.assertRaises(lsh_matmul([1, 2], [1, 2, 3, 4]).eval(), ValueError)
            with self.assertRaises(ValueError):
                self.assertRaises(lsh_matmul([1, 2, 3], [[1, 2], [3, 4]]).eval(), ValueError)
    '''
    def test_lshMatMulHardCoded(self):
        with self.session():
            result = lsh_matmul([[1], [2]], [[1, 2], [3, 4]]).numpy()
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result[0], 5)
            self.assertEqual(result[1], 11)
    '''
    def test_lshMatMulGradientXHardCoded(self):
        with self.session() as sess:
            x = tf.placeholder(tf.float32, shape = (2))
            W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_lsh_matmul = lsh_matmul(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_lsh_matmul = tf.gradients(Wx_lsh_matmul, x)
            
            gradient_tf = sess.run(grad_x_tf, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            gradient_lsh_matmul = sess.run(grad_x_lsh_matmul, feed_dict = {x: np.asarray([1, 2]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0], gradient_lsh_matmul[0][0])
            self.assertEqual(gradient_tf[0][1], gradient_lsh_matmul[0][1])
    
    def test_lshMatMulGradientWHardCoded(self):
        with self.session() as sess:
            x = tf.constant(np.asarray([1, 2]).astype(np.float32))
            W = tf.placeholder(tf.float32, shape = (2, 2))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_lsh_matmul = lsh_matmul(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_lsh_matmul = tf.gradients(Wx_lsh_matmul, W)
            
            gradient_tf = sess.run(grad_W_tf, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            gradient_lsh_matmul = sess.run(grad_W_lsh_matmul, feed_dict = {W: np.asarray([[1, 2], [3, 4]]).astype(np.float32)})
            
            self.assertEqual(gradient_tf[0][0][0], gradient_lsh_matmul[0][0][0])
            self.assertEqual(gradient_tf[0][0][1], gradient_lsh_matmul[0][0][1])
            self.assertEqual(gradient_tf[0][1][0], gradient_lsh_matmul[0][1][0])
            self.assertEqual(gradient_tf[0][1][1], gradient_lsh_matmul[0][1][1])
    
    def test_lshMatMulRandom(self):
        with self.session():
            n = 4
            m = 5
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n, 1))
                W_rand = np.random.randint(10, size = (m, n))
                result_rand = np.dot(W_rand, x_rand)
                
                result = lsh_matmul(x_rand, W_rand).eval()
                np.testing.assert_array_equal(result, result_rand)
    
    def test_lshMatMulGradientXRandom(self):
        with self.session() as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_lsh_matmul = lsh_matmul(tf.reshape(x, [-1, 1]), W)
            
            grad_x_tf = tf.gradients(Wx_tf, x)
            grad_x_lsh_matmul = tf.gradients(Wx_lsh_matmul, x)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_x_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_lsh_matmul = sess.run(grad_x_lsh_matmul, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_lsh_matmul)
                
    def test_lshMatMulGradientWRandom(self):
        with self.session() as sess:
            n = 4
            m = 5
            
            x = tf.placeholder(tf.float32, shape = (n))
            W = tf.placeholder(tf.float32, shape = (m, n))
            
            Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))
            Wx_lsh_matmul = lsh_matmul(tf.reshape(x, [-1, 1]), W)
            
            grad_W_tf = tf.gradients(Wx_tf, W)
            grad_W_lsh_matmul = tf.gradients(Wx_lsh_matmul, W)
            
            for i in range(100):
                x_rand = np.random.randint(10, size = (n))
                W_rand = np.random.randint(10, size = (m, n))
                
                gradient_tf = sess.run(grad_W_tf, feed_dict = {x: x_rand, W: W_rand})
                gradient_lsh_matmul = sess.run(grad_W_lsh_matmul, feed_dict = {x: x_rand, W: W_rand})
                
                np.testing.assert_array_equal(gradient_tf, gradient_lsh_matmul)
                  
    '''       
if __name__ == '__main__':
    test.main()
