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
        # SAX Bin'lerini hesapla (Global percentiles)
        self.bins = np.percentile(ts, np.linspace(0, 100, self.alphabet_size + 1)[1:-1])
        
        # Sliding Window + PAA + SAX ile örüntüleri çıkar
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
            
        # Olasılıkları hesapla: P(Si -> Sj) = Geçiş Sayısı / Toplam Çıkış Sayısı
        for state, next_states in transition_counts.items():
            self.transitions[state] = {}
            for n_state, count in next_states.items():
                self.transitions[state][n_state] = count / out_counts[state]

    def _paa(self, subsequence, num_segments):
        """Piecewise Aggregate Approximation (PAA) Algoritması"""
        n = len(subsequence)
        w = num_segments
        if n == w:
            return subsequence
            
        if n % w == 0:
            return np.mean(subsequence.reshape(w, -1), axis=1)
        else:
            paa_res = np.zeros(w)
            for i in range(w):
                start = i * n / w
                end = (i + 1) * n / w
                start_idx = int(np.floor(start))
                end_idx = int(np.ceil(end))
                
                if end_idx == start_idx + 1:
                    paa_res[i] = subsequence[start_idx]
                else:
                    seg = subsequence[start_idx:end_idx]
                    weights = np.ones(len(seg))
                    weights[0] = start_idx + 1 - start
                    weights[-1] = end - (end_idx - 1)
                    paa_res[i] = np.sum(seg * weights) / (n / w)
            return paa_res

    def _sax(self, paa_sequence):
        """Symbolic Aggregate approXimation (SAX) Algoritması"""
        indices = np.digitize(paa_sequence, self.bins)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return "".join([alphabet[i] for i in indices])

    def _extract_patterns(self, ts):
        """Sliding Window ile zaman serisinden örüntü çıkartır"""
        patterns = []
        n = len(ts)
        sliding_window_length = self.window_size 
        paa_segments = self.window_size
        
        for i in range(n - sliding_window_length + 1):
            subsequence = ts[i : i + sliding_window_length]
            
            # 1. PAA Dönüşümü (Boyut indirgeme / yumuşatma)
            paa_seq = self._paa(subsequence, paa_segments)
            
            # 2. SAX Dönüşümü (Sembolik temsile çevirme)
            pattern = self._sax(paa_seq)
            
            patterns.append(pattern)
            
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
        """Olasılıksal Açıklanabilirlik Modülü Çıktısı (JSON Formatı)"""
        status = "seen"
        mapped_to = incoming_pattern
        
        # Unseen Pattern Yönetimi
        if incoming_pattern not in self.vocabulary:
            status = "unseen"
            mapped_to = self._get_nearest_pattern(incoming_pattern)
            
        # Geçiş Olasılığı (Confidence Score / Path Probability)
        prob = self.transitions.get(prev_state, {}).get(mapped_to, 0.0)
        
        # Anomali Kararı (Eşik Değeri Örneği: %5)
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