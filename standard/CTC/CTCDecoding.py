import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        C, T, B = y_probs.shape
        probs = y_probs[:, :, 0]
        for t in range(T):
            c = int(np.argmax(probs[:, t]))
            path_prob *= float(probs[c, t])
            decoded_path.append(c)

        tokens = []
        prev_tok = None
        for c in decoded_path:
            if c == 0:
                prev_tok = None
                continue
            tok = self.symbol_set[c - 1]
            if tok != prev_tok:
                tokens.append(tok)
            prev_tok = tok

        decoded_path = "".join(tokens)
    
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width


    def decode(self, y_probs):
        """
        Perform beam search decoding (CTC-style)
        """
        C, T, B = y_probs.shape
        probs = y_probs[:, :, 0]
        BLANK = 0
        ActivePaths = {("", True): 1.0}
        TempPaths = {}
        for t in range(T):
            symbol_probs = probs[:, t]
            top_active = sorted(ActivePaths.items(), key=lambda kv: kv[1], reverse=True)[: self.beam_width]
            TempPaths = {}
            for (path, last_blank), path_score in top_active:
                new_path = path
                new_score = path_score * float(symbol_probs[BLANK])
                state = (new_path, True)
                if state in TempPaths:
                    TempPaths[state] += new_score
                else:
                    TempPaths[state] = new_score
                for c in range(1, C):
                    ch = self.symbol_set[c - 1]
                    new_path = path if (not last_blank and len(path) > 0 and path[-1] == ch) else path + ch
                    new_score = path_score * float(symbol_probs[c])
                    state = (new_path, False)
                    if state in TempPaths:
                        TempPaths[state] += new_score
                    else:
                        TempPaths[state] = new_score
            ActivePaths = TempPaths
        MergedPathScores = {}
        for (path, _), score in ActivePaths.items():
            trimmed = path.strip()
            if trimmed in MergedPathScores:
                MergedPathScores[trimmed] += score
            else:
                MergedPathScores[trimmed] = score
        BestPath = "" if not MergedPathScores else max(MergedPathScores.items(), key=lambda kv: kv[1])[0]
        MergedPathScores = {k: np.array([v]) for k, v in MergedPathScores.items()}
        return BestPath, MergedPathScores



