import tensorflow as tf


class RandSRPInitializer(tf.keras.initializers.Initializer):

def __init__(self):

def __call__(self, shape, dtype=None):
  #
  
  #return tf.random.normal(shape, dtype=dtype)
  #int *a = new int[_dim];
  # for (size_t i = 0; i < _dim; i++) {
    # a[i] = i;
  # }
  # // i, j, k == a[i][j][k] == randBits[l*k][s]
  # srand(time(0));
  # _randBits = new short *[_numhashes]; //k*L*samSize!!
  # _indices = new int *[_numhashes];
  # // please explain the hashing technique
    # for (size_t i = 0; i < _numhashes; i++) {
        # random_shuffle(a, a+_dim); // permutation (0,1,...n-1) 
        # _randBits[i] = new short[_samSize];
        # _indices[i] = new int[_samSize];
        # for (size_t j = 0; j < _samSize; j++) {
            # _indices[i][j] = a[j];
            # int curr = rand();
            # if (curr % 2 == 0) {
                # _randBits[i][j] = 1;
            # } else {
                # _randBits[i][j] = -1;
            # }
        # }
        
        # // why are you sorting again?
        # // did you mean to sort a?
        # std::sort(_indices[i], _indices[i]+_samSize);
    # }
    # delete [] a;

def get_config(self):  # To support serialization
  return {'mean': self.mean, 'stddev': self.stddev}