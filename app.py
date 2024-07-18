
import tkinter as tk
from tkinter import Label, Button, Toplevel, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json
from textblob import Word
import time
from labels import labels

# Load model architecture from JSON file
with open("ISLRS\SLModel.json", "r") as json_file:
    model_json = json_file.read()

# Load model from JSON
model = model_from_json(model_json)

# Load weights into the model
model.load_weights("ISLRS\SLModel.h5")

# Label list
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#           'W', 'X', 'Y', 'Z', 'blank']

myLabels = labels

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def show_about():
    about_window = Toplevel()
    about_window.title("About")
    about_label = Label(about_window, text="Sign Language Recognition System\nVersion 1.0\nDeveloped by Gautam Chhajyan, Aman Kumar, Arpit Katiyar")
    about_label.pack(padx=10, pady=10)

def get_suggestions(text):
    word = Word(text)
    suggestions = word.spellcheck()[:3]  # Get the top 3 suggestions
    return [suggestion[0] for suggestion in suggestions]

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize variables for current word and sentence
        self.current_word = ""
        self.sentence = ""
        self.last_add_time = time.time()
        self.first_symbol_detected = False  # Flag to track the first symbol detection

        # Create about button
        self.btn_about = Button(window, text="About", command=show_about)
        self.btn_about.place(relx=0.0, x=10, y=10, anchor='nw')

        self.btn_quit = Button(window, text="Quit", command=window.quit)
        self.btn_quit.place(relx=1.0, x=-10, y=10, anchor='ne')

        # Add the title of the project
        self.project_title = Label(window, text="Sign Language Recognition System", font=("Helvetica", 20, "bold"))
        self.project_title.pack(pady=10)

        # Create a frame for the camera
        self.video_frame = Frame(window)
        self.video_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, width=300, height=300)
        self.canvas.pack()

        self.label_prediction = Label(window, text="Prediction: ", font=("Helvetica", 16))
        self.label_prediction.pack(pady=10)

        self.label_word = Label(window, text="Word: ", font=("Helvetica", 16))
        self.label_word.pack(pady=10)

        # Create a frame for the suggestion buttons
        self.suggestion_frame = Frame(window)
        self.suggestion_frame.pack(pady=5)

        self.suggestion_buttons = []
        for _ in range(3):
            btn = Button(self.suggestion_frame, text="", font=("Helvetica", 16), command=lambda i=_: self.on_suggestion_click(i))
            btn.pack(side=tk.LEFT, padx=5)
            self.suggestion_buttons.append(btn)

        # Create a frame for word and sentence management buttons
        self.word_sentence_frame = Frame(window)
        self.word_sentence_frame.pack(pady=10)

        self.btn_add_word = Button(self.word_sentence_frame, text="Add Word to Sentence", command=self.add_word_to_sentence)
        self.btn_add_word.pack(side=tk.LEFT, padx=5)

        self.btn_clear_sentence = Button(self.word_sentence_frame, text="Clear Sentence", command=self.clear_sentence)
        self.btn_clear_sentence.pack(side=tk.LEFT, padx=5)

        self.label_sentence = Label(window, text="Sentence: ", font=("Helvetica", 16))
        self.label_sentence.pack(pady=10)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.delay = 50
        self.update()
        self.window.mainloop()

    def on_suggestion_click(self, index):
        suggestion = self.suggestion_buttons[index].cget("text")
        self.current_word = suggestion
        self.label_word.config(text=f'Word: {self.current_word}')
        self.add_word_to_sentence()



    def add_word_to_sentence(self):
        # Append current word to sentence and clear current word
        if self.current_word:
            self.sentence += self.current_word + " "
            self.label_sentence.config(text=f'Sentence: {self.sentence}')
            self.current_word = ""
            self.label_word.config(text='Word: ')

    def clear_sentence(self):
        # Clear the sentence
        self.sentence = ""
        self.label_sentence.config(text='Sentence: ')

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            cropframe = frame[40:300, 0:300]
            cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
            cropframe = cv2.resize(cropframe, (48, 48))
            cropframe = extract_features(cropframe)

            # Predict the class
            pred = model.predict(cropframe)
            prediction_label = myLabels[pred.argmax()]

            # Display the prediction
            accu = "{:.2f}".format(np.max(pred) * 100)
            self.label_prediction.config(text=f'Prediction: {prediction_label}  {accu}%')

            # Update current word with time gap
            current_time = time.time()
            if not self.first_symbol_detected:
                # For the first symbol, wait for some time before predicting
                if (current_time - self.last_add_time) > 6:  # 5 seconds gap
                    if prediction_label != 'blank':  # If the first symbol is not blank
                        self.current_word += prediction_label
                        self.label_word.config(text=f'Word: {self.current_word}')
                    self.last_add_time = current_time
                    self.first_symbol_detected = True  # Set flag to indicate first symbol detected
            else:
                if (current_time - self.last_add_time) > 4:  # 5 seconds gap
                    if prediction_label != 'blank':  # If the predicted symbol is not blank
                        self.current_word += prediction_label
                        self.label_word.config(text=f'Word: {self.current_word}')
                    else:  # If the predicted symbol is blank
                        if self.current_word:  # If current word is not empty, add it to sentence
                            self.sentence += self.current_word + " "
                            self.label_sentence.config(text=f'Sentence: {self.sentence}')
                            self.current_word = ""  # Reset current word
                            self.label_word.config(text='Word: ')
                    self.last_add_time = current_time

            # Example text input for suggestions
            text_input = self.current_word.lower()
            suggestions = get_suggestions(text_input)

            for btn, suggestion in zip(self.suggestion_buttons, suggestions):
                btn.config(text=suggestion)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)



App(tk.Tk(), "Sign Language Recognition")