ROW_LENGTH = 50
NUMBER_OF_ROWS = 21
INPUT_LAYER_SIZE = ROW_LENGTH * NUMBER_OF_ROWS

ALPHABET = ['ᡠ',
            'ᡬ',
            'ᡴ',
            'ᡦ',
            'ᡤ',
            'ᡮ',
            'ᠮ',
            'ᠰ',
            'ᠵ',
            'ᠨ',
            'ᡳ',
            'ᡟ',
            'ᠸ',
            'ᡧ',
            'ᡵ',
            'ᠪ',
            '᠈',
            'ᠶ',
            '᠉',
            'ᠩ',
            'ᠠ',
            'ᠴ',
            'ᠯ',
            'ᡝ',
            'ᡷ',
            'ᡰ',
            'ᡥ',
            'ᠺ',
            'ᠣ',
            'ᡭ',
            'ᡱ',
            'ᡨ',
            'ᡯ',
            'ᡩ',
            'ᡶ',
            '\'᠋',
            'ᡡ',
            ' ',    # whitespace
            '*']    # blank
OUTPUT_LAYER_SIZE = len(ALPHABET)

DB_NAME = "./manchu_transliteration.db"
