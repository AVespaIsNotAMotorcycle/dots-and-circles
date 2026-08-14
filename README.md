# About

Dots and Circles is my first non-tutorial machine learning project. It is an app for optical character recognition of Manchu-language text. Since the aim was to further my understanding of neural networks, I opted not to use a framework like PyTorch, and instead implemented the neural networks using numpy matrices.

# The Problem

The Manchu script is written top-to-bottom, left-to-right. All letters within a word are connected to each other, and the shape of a letter changes based on its position within a word. This makes it quite difficult to break a word down into individual letters. In order to circumvent this, I decided to procede using a basic form of connectionist temporal classification (CTC).

# The Approach

Training data was generated synthetically using the [manchu-cake](https://github.com/OverflowCat/manchu-cake) dictionary. Each unique manchu word in the dictionary was stored in a SQLite database, and random words were pulled from the database and rendered in a random font. Each image of a word in a particular font (called a **lexigraph** from here on) had letter boundaries manually labelled. Each lexigraph is 50 pixels wide, and there are 10 pixels of whitespace on each side.

As there are various styles of written Manchu, a lexigraph is first sorted into one of four **classes**. I defined the classes by comparing three features in various Manchu fonts: line weight, whether the letters romanized as "m" and "l" are connected to or disconnected from the center line, and whether the letters romanized as "a" and "e" are pointy or rounded. The actual classification is performed by an RNN, which I call the **Classifier**, which reads the image line by line.

OCR is performed by two neural networks, each with four instances, one for each lexigraph class. For example, a lexigraph of class A is analyzed by a neural network trained only on class A lexigraphs, likewise for classes B, C, and D. The first networks, henceforth called the **Primary OCR**, takes as input 21 sequential rows of pixels in a lexigraph and outputs a single character. This character is what the Primary OCR thinks the middle row of pixels corresponds to. The Primary OCR has two hidden layers.

Unfortunately, the Primary OCR had difficulty differentiating certain letters. However, it tended to produce errors in patterns. In order to counteract this, I added a second network, called the **Secondary OCR**, which took as input 21 outputs from the Primary OCR and output the character it predicted the middle to actually correspond to. This had the effect of reducing noise in output and marginally improving accuracy.

Finally, the output of the Secondary OCR is parsed and rendered as Unicode text. This is done on the [frontend](https://github.com/AVespaIsNotAMotorcycle/dots-and-circles-frontend). The server sends the frontend the Secondary OCR's output as an array of characters, which may be any character in the Manchu Unicode block (a subset of the Mongolian block), a " " character, or a "*". A " " represents a pixel thought to be whitespace, and a "*" represents a break between two characters. To minimize the effect of noisy data, any characters which have no identical neighbors are deleted. For example, in the array ['a', 'a', 'b', 'a'], 'b' would be deleted because neither of its neighbors match it. Once this is done, all identical neighboring characters are collapsed into one. For example, ['a', 'a', 'b', 'b', 'b', 'c'] becomes ['a', 'b', 'c']. At this point, any '*' characters are removed. Their removal after the collapsing of neighboring characters allows the program to account for two letters in sequence. For example, ['a', 'a', '*', 'a'] becomes ['a', 'a']. At this point the array is joined to create a string as the final output.

# Lessons Learned

At some point in the future I aim to redo this project because, let's be real, it's quite inaccurate. When I do, I will likely use an encoder/decoder model, with an RNN (the encoder) first reading the lexigraph line by line and producing some representation of the word, and a second RNN (the decoder) using that representation to spell out the word character by character.

Additionally, I abandon the marking of character boundaries in favor of giving the machine whole lexigraphs and allowing it to learn the boundaries on its own. This would make it easier to generate training data, and make it easier to incorporate real-world data. There is, for example, a publically available [dataset](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586) of words scanned from Manchu books published by Sun Haipeng, Tao Wenhao, and Bi Xiaojun. In this future version, I will use a test set composed entirely of real-world data, while the training set would be a mix of synthetic and real-world lexigraphs. Using whole words rather than manually labelled characters would make it much easier to assemble a large body of training data. As of the time of writing, there are 166 lexigraphs in the training data, which is quite a small training set.

Another change I would like to make in a future version is to add positional encoding. Right now, the Primary OCR's only hint to what part of the word it's looking at is the slice of the image it's given. It may be helpful to have another input which tells it whether that slice is at the beginning, end, or middle of a word.
