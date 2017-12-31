# Optical-Character-Recognition-using-Hidden-Markov-Models


Using the versatility of HMMs, let’s try applying them to another problem; Optical Character Recognition. Our goal is to recognize
text in an image where the font and font size is known ahead of time. Modern OCR is very good at recognizing documents, but rather poor when recognizing isolated characters.
It turns out that the main reason for OCR’s success is that there’s a strong language model: the algorithm
can resolve ambiguities in recognition by using statistical constraints of English (or whichever language is
being processed). These constraints can be incorporated very naturally using an HMM.</br> </br>

__Data:__ A text string image is divided into little subimages corresponding to individual letters;
a real OCR system has to do this letter segmentation automatically, but here we’ll assume a fixed-width
font so that we know exactly where each letter begins and ends ahead of time. In particular, we’ll assume
each letter fits in a box that’s 16 pixels wide and 25 pixels tall. We’ll also assume that our documents only
have the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation
symbols, (),.-!?’". Suppose we’re trying to recognize a text string with n characters, so we have n observed
variables (the subimage corresponding to each letter) O 1 , ..., O n and n hidden variables, l 1 ..., l n , which are
the letters we want to recognize. We’re thus interested in P (l 1 , ..., l n |O 1 , ..., O n ). We can rewrite
this using Bayes’ Law, estimate P (O i |l i ) and P (l i |l i−1 ) from training data, then use probabilistic inference
to estimate the posterior, in order to recognize letters. </br> </br>

The program loads the image file, which contains images of letters to use for training. It also loads the text training file, which is simply some text document that
is representative of the language (English, in this case) that will be recognized. Then, it uses the classifier it has learned to detect the text in test-image-
file.png, using (1) simple Bayes net (2) Hidden Markov Model with variable elimination, and
(3) Hidden Markov Model with MAP inference (Viterbi). The output displays the text as recognized using the above three approaches.

The program is called like this:
__./ocr.py train-image-file.png train-text.txt test-image-file.png__

