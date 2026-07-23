## Strategy
All words are stored as 50x350 images. We're gonna break that down into 50x5 chunks, then feed in
3 chunks in sequence. Each time we do this we are advancing by one chunk, as visualized below:

        Step 1  Step 2  Step 3
Chunk 0 ||||||
Chunk 1 ||||||  ||||||  
Chunk 2 ||||||  ||||||  ||||||
Chunk 3         ||||||  ||||||
Chunk 4                 ||||||

Each time we do this we will generate guesses as to what letter the middle chunk represents.

For this we will need training data which has labelled chunks.

Once the NN has looked at all the chunks in a word and given its guesses as to the letter, it will
have generated a series of responses with more responses than actual letters. This can be collapsed
like m m m a a n n j j u u => m a n j u. One possible issue is when the same letter actually is
present twice in a row - CTC uses blank outputs to represent gaps between two of the same letter.
We'll have to label some chunks as blanks - any which are just center line I suppose.

Once all the chunks have been labelled and collapsed etc we may want to check a dictionary to see
whether that word, or a very similar one, actually exists.

## Training Data Generation
Frontend - Data labelling:
I think I want a nice interface for this, so I'll build a little front-end.
Press a button, and you are presented with a random word, rendered as an image, with marked chunks.
There's a form for labelling them. Upon submission, the word image is saved in the backend with a
couple variations, including blurred or smudged ones? Maybe some which are slightly curved rather
than having a perfectly straight center line. However, since font can significantly change the
length of a word, chunk labelling can only apply to one font/word combination. The same word in a
different font would require separate labelling.

