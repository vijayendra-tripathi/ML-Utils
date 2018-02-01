# ML-Utils
Useful collection of machine learning helper classes and functions.

Python classes:
===============

ML_Utils.py
-----------
This is Keras Callback subclass created to write output to a file after every epoch finishes. This is handy if you loose internet connection to a remote server, especially while running training on cloud. The server will write output after every epoch to a file. You can recover all logs after training to see what happened in the background.
