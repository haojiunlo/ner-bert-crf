import unicodedata, re
from bert4keras.snippets import is_py2
from bert4keras.tokenizers import Tokenizer

class CustomTokenizer(Tokenizer):
    
    # tokenize space to a special token: [unused1]
    def _tokenize(self, text):
        if self._do_lower_case:
            if is_py2:
                text = unicode(text)
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([
                ch for ch in text if unicodedata.category(ch) != 'Mn'
            ])

        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += " [unused1] "
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))
        
        return tokens
    
    
    def rematch(self, text, tokens):
        if is_py2:
            text = unicode(text)

        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        text_ = ""
        for ch in text:
            if self._is_space(ch):
                ch = " "
            text_ += ch

        for token in tokens:
            if token == "[unused1]":
                start = text_[offset:].index(" ") + offset
                end = start + 1
                token_mapping.append(char_mapping[start:end])
                offset = end
            elif self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text_[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping
    