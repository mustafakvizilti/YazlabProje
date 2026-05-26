import numpy as np
import json

class ProbabilisticAutomata:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.transitions = {}
        self.vocabulary = set()
        self.bins = None

    def fit(self, ts):
        self.bins = np.percentile(ts, np.linspace(0, 100, self.alphabet_size + 1)[1:-1])
        patterns = self._extract_patterns(ts)
        self.vocabulary = set(patterns)
        
        transition_counts = {}
        out_counts = {}
        
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i+1]
            
            if current_state not in transition_counts:
                transition_counts[current_state] = {}
                out_counts[current_state] = 0
                
            if next_state not in transition_counts[current_state]:
                transition_counts[current_state][next_state] = 0
                
            transition_counts[current_state][next_state] += 1
            out_counts[current_state] += 1
            
        for state, next_states in transition_counts.items():
            self.transitions[state] = {}
            for n_state, count in next_states.items():
                self.transitions[state][n_state] = count / out_counts[state]

    def _extract_patterns(self, ts):
        indices = np.digitize(ts, self.bins)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        chars = [alphabet[i] for i in indices]
        patterns = []
        for i in range(len(chars) - self.window_size + 1):
            patterns.append("".join(chars[i:i+self.window_size]))
        return patterns

    def _levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dele = curr[j] + 1
                sub = prev[j] + (c1 != c2)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[-1]

    def _get_nearest_pattern(self, pattern):
        min_dist = float('inf')
        best_match = pattern
        for vocab_pattern in self.vocabulary:
            dist = self._levenshtein(pattern, vocab_pattern)
            if dist < min_dist:
                min_dist = dist
                best_match = vocab_pattern
        return best_match

    def explain_step(self, prev_state, incoming_pattern, time_step):
        status = "seen"
        mapped_to = incoming_pattern
        
        if incoming_pattern not in self.vocabulary:
            status = "unseen"
            mapped_to = self._get_nearest_pattern(incoming_pattern)
            
        prob = self.transitions.get(prev_state, {}).get(mapped_to, 0.0)
        
        decision = "anomaly" if prob < 0.05 else "normal"
        
        explanation = {
            "time_step": time_step,
            "state": prev_state,
            "pattern": incoming_pattern,
            "status": status,
            "mapped_to": mapped_to,
            "probability": prob,
            "decision": decision
        }
        return explanation