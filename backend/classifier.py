class Classifier:
    def __init__(self): return

    def classify(self, font):
        classA = [0, 1, 4, 7, 9, 10]
        classB = [2, 5]
        classC = [3, 8]
        classD = [6]

        if int(font) in classA: return 'A'
        if int(font) in classB: return 'B'
        if int(font) in classC: return 'C'
        return 'D'
