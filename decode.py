import csv
import numpy as np
import string
import time
# np.random.seed(seed=12)

'''
* Work with log probabilities to sum instead of multiply.

Speedups:
* Remove the first element of the likelihood expression.
* Store the recent likelihood to not compute on each cycle. x2 speedup.
* Acceptance criterion is random < log-likelihood(f_(y)) - log_likelihood(f(y))
* Store log transition probabilities. x3 speedup.
* (?) Faster convergence, do not try a function twice?
    Yes. Memoize the function and map to transition counts.

* Compute transition counts for each function and then only compute the acceptance criterion.
'''
'''
class DecoderSlow():
    def __init__(self, y_symbols):
        # Alphabet.
        self.alphabet = list(string.ascii_lowercase) + [" ", "."]
        self.symbol_to_id = dict(map(reversed, enumerate(self.alphabet)))
        self.id_to_symbol = dict(enumerate(self.alphabet))
        
        # Transition probabilities.
        self.M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',', dtype = np.float64)
        self.M = np.log(self.M)
        self.P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',', dtype = np.float64)

        # Turn y to array of indexes.
        self.y = np.array([self.symbol_to_id[s] for s in y_symbols])

    def decode(self, f = None):
        """ Run the decode loop to convergence.

        Args:
            y (np.array (1 x n)): The encrypted string where each element is an id for a symbol.
            f (np.array (1 x m)): The mapping between symbol indexes to decrypted symbol.
        """
        # If no mapping f given, then start with a random one.
        if f is None:
            f = np.array([i for i in range(len(self.alphabet))])
            np.random.shuffle(f)
        # Sequence decoded with f.
        x_hat_f = f[self.y]
        log_like_f = self.log_likelihood(x_hat_f )

        best_log_like = -float('inf')
        x_hat_best = np.array(x_hat_f)

        accepted = 0
        for iter in range(8000):

            # Get a candidate mapping by swapping two elements.
            f_ = np.array(f)
            idx = range(len(f_))
            ix1, ix2 = np.random.randint(len(self.alphabet), size=2) # This is also the length of f and f_.
            f_[ix1], f_[ix2] = f_[ix2], f_[ix1]


            # Decode with both candidates.
            # x_hat_f = f[self.y]
            x_hat_f_ = f_[self.y]

            # Get log likelihood.
            # log_like_f = self.log_likelihood(x_hat_f )
            log_like_f_= self.log_likelihood(x_hat_f_)


            # Ratio between the likelihoods.
            # like_ratio = np.exp(log_like_f_ - log_like_f)
            log_like_ratio = log_like_f_ - log_like_f

            if np.log(np.random.rand()) < log_like_ratio or np.isnan(log_like_ratio):
                if log_like_f_ > best_log_like:
                    best_log_like = log_like_f_
                    x_hat_best = x_hat_f_

                f = np.array(f_)
                log_like_f = log_like_f_
                accepted+=1

        # print("accepted", accepted)
        # Turn back to letters and join.
        x_hat_symbols = "".join([self.id_to_symbol[i] for i in x_hat_best])
        return x_hat_symbols

    def log_likelihood(self, x_hat):

        # log_like = np.log(self.P[x_hat[0]]) + np.sum(np.array([np.log(self.M[x_hat[i]][x_hat[i-1]])  for i in range(2, len(x_hat))])) 
        log_like =  np.sum(np.array([self.M[x_hat[i]][x_hat[i-1]]  for i in range(1, len(x_hat))])) 
        return log_like
'''




class Decoder():
    def __init__(self, y_symbols):
        # Alphabet.
        self.alphabet = list(string.ascii_lowercase) + [" ", "."]
        self.symbol_to_id = dict(map(reversed, enumerate(self.alphabet)))
        self.id_to_symbol = dict(enumerate(self.alphabet))
        
        # Transition probabilities.
        self.M = np.loadtxt('./data/letter_transition_matrix.csv', delimiter=',', dtype = np.float64)
        eps = 1e-308
        self.M = np.log(self.M + eps)
        self.P = np.loadtxt('./data/letter_probabilities.csv', delimiter=',', dtype = np.float64)

        # Set up Transition Count matrix TC.
        self.TC = np.zeros((28,28))
        for i in range(1, len(y_symbols)):
            self.TC[self.symbol_to_id[y_symbols[i]]][self.symbol_to_id[y_symbols[i-1]]] += 1
        

        # Turn y to array of indexes.
        self.y = np.array([self.symbol_to_id[s] for s in y_symbols])

    def update_y_symbols(self, y_symbols):
        # Set up Transition Count matrix TC.
        self.TC = np.zeros((28,28))
        for i in range(1, len(y_symbols)):
            self.TC[self.symbol_to_id[y_symbols[i]]][self.symbol_to_id[y_symbols[i-1]]] += 1
        

        # Turn y to array of indexes.
        self.y = np.array([self.symbol_to_id[s] for s in y_symbols])

    def decode(self, f = None, iters = 6000):
        """ Run the decode loop to convergence.

        Args:
            y (np.array (1 x n)): The encrypted string where each element is an id for a symbol.
            f (np.array (1 x m)): The mapping between symbol indexes to decrypted symbol.
        """
        # If no mapping f given, then start with a random one.
        if f is None:
            f = np.array([i for i in range(len(self.alphabet))])
            np.random.shuffle(f)
        
        # The current Transition Count matrix. For f().
        range28 = np.arange(28)
        TC_f = np.array(self.TC)
        TC_f[f] = TC_f[range28]
        TC_f.T[f] = TC_f.T[range28]

        # Memoize. Store the Transition Count matrix for each permutation f().
        memo = dict()
        memo[hash(f.tostring())] = TC_f

        accepted = 0
        for iter in range(iters):
        # while accepted < 2000:
            # Choose two ixs from f.
            f_ = np.array(f)
            idx = range(len(f_))
            ix1, ix2 = np.random.randint(len(self.alphabet), size=2) # This is also the length of f and f_.
            # These ixs are switched in f_.
            f_[ix1], f_[ix2] = f_[ix2], f_[ix1]

            h_f_ = hash(f_.tostring())
            if h_f_ in memo:
                TC_f_ = memo[h_f_]

            else:
                # Permute the transition matrices to represent the permutations.
                TC_f_ = np.array(self.TC)
                
                TC_f_[f_] = TC_f_[range28]
                TC_f_.T[f_] = TC_f_.T[range28]
                memo[h_f_] = TC_f_

            # The chosen ids that have been switched.
            a = f[ix1]
            b = f[ix2]

            crit = TC_f_[a, :].dot(self.M[a, :]) + \
                   TC_f_[b, :].dot(self.M[b, :]) + \
                   TC_f_[:, a].dot(self.M[:, a]) + \
                   TC_f_[:, b].dot(self.M[:, b]) - \
                   TC_f[a, :].dot(self.M[a, :]) - \
                   TC_f[b, :].dot(self.M[b, :]) - \
                   TC_f[:, a].dot(self.M[:, a]) - \
                   TC_f[:, b].dot(self.M[:, b])
            

            # crit = np.sum(np.multiply(TC_f_, self.M) - np.multiply(TC_f, self.M))

            u = np.log(np.random.rand())
            if u < crit: # or np.isnan(crit):
                
                f = f_
                TC_f = TC_f_
                accepted+=1

        # Turn back to letters and join.
        # print("accept", accepted)
        x_hat = f[self.y]
        x_hat_symbols = "".join([self.id_to_symbol[i] for i in x_hat])
        log_like = np.sum(np.multiply(TC_f, self.M))
        return x_hat_symbols, f, log_like


    def log_likelihood(self, x_hat):

        # log_like = np.log(self.P[x_hat[0]]) + np.sum(np.array([np.log(self.M[x_hat[i]][x_hat[i-1]])  for i in range(2, len(x_hat))])) 
        log_like =  np.sum(np.array([self.M[x_hat[i]][x_hat[i-1]]  for i in range(1, len(x_hat))])) 
        return log_like

    def f_to_text(self, f):
        x_hat = f[self.y]
        return "".join([self.id_to_symbol[i] for i in x_hat])

def decode_no_breakpoint(y_symbols, decode_iters):
    # Strategy.
    # Just try it out many times and take the one with the highest likelihood.
    # Start time.
    start_time = time.time()

    # Variables to store best values.
    best_overall_log_like = -np.inf
    best_plaintext = ""
    # Create decoder.
    decoder = Decoder(y_symbols)

    time_elapsed = 0
    iter = 0
    num_iters = 3
    while iter < num_iters and time_elapsed < 110:
        iter += 1
        # Choose split ix.
        plaintext, f, log_like = decoder.decode(iters = decode_iters)

        # Check if overall likelihood is better than the one seen so far.
        if best_overall_log_like < log_like:
            best_overall_log_like = log_like
            best_plaintext = plaintext

        # How much time do we have left.
        time_elapsed = time.time() - start_time 
    return best_plaintext

def decode_breakpoint(y_symbols, decode_iters):
    # Strategy:
    # Split the text to 2 parts.
    # Instanatiate a Decoder for each.
    # Run decode on each part.
    # Get the likelihood for each side, combine to get the best likelihood so far.
    # After some iterations, return the decoded text.

    # Start time.
    start_time = time.time()

    # Variables to store best values.
    best_overall_log_like = -np.inf
    best_plaintext = y_symbols
    # Create decoder for left side.
    decoder_left = Decoder("this is dummy text.")
    # Create decoder for right side.
    decoder_right= Decoder("this is dummy text.")
    time_elapsed = 0
    iter = 0
    num_iters = 19
    while time_elapsed < 110 and iter < num_iters:
        iter += 1
        # for iter in range(60) and time_elapsed < 110:
        # Choose split ix.
        split_ix = iter * int(len(y_symbols)//(num_iters))
        y_symbols_left, y_symbols_right = y_symbols[:split_ix], y_symbols[split_ix:]
        
        # Run left side.
        # decoder_left = Decoder(y_symbols_left)
        decoder_left.update_y_symbols(y_symbols_left)
        plaintext_left, f_left, log_like_left = decoder_left.decode(iters = decode_iters)

        # Run left side.
        # decoder_right = Decoder(y_symbols_right)
        decoder_right.update_y_symbols(y_symbols_right)
        plaintext_right, f_right, log_like_right = decoder_right.decode(iters = decode_iters)

        # Check if overall likelihood is better than the one seen so far.
        if best_overall_log_like < log_like_left + log_like_right:
            best_overall_log_like = log_like_left + log_like_right
            best_plaintext = plaintext_left + plaintext_right

        # How much time do we have left.
        time_elapsed = time.time() - start_time 

        # TODO: Consider ranking estimated texts by letter occurance frequency.
    return best_plaintext

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    if not has_breakpoint:
        # Instance of decoder.
        plaintext = decode_no_breakpoint(ciphertext, decode_iters= 5000)
        return plaintext
    else:
        plaintext = decode_breakpoint(ciphertext, decode_iters=5000)
        return plaintext