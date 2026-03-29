import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2
from keras.models import load_model

model = load_model("Alphabet_Recognition.h5")
alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class SketchPad(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Alphabet Classifier - Sketchpad")
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a letter", font=("Helvetica", 22))
        self.classify_btn = tk.Button(self, text="Predict", command=self.classify_handwriting)
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear_canvas)

        self.canvas.grid(row=0, column=0, pady=2, padx=2)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=0, pady=2, padx=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a letter")

    def preprocess(self, img):
        img = np.array(img)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(img)

        coords = cv2.findNonZero(img)
        if coords is None:
            return np.zeros((28, 28), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w, new_h = 20, int(20 / aspect_ratio)
        else:
            new_h, new_w = 20, int(20 * aspect_ratio)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        kernel = np.ones((2, 2), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)

        return canvas

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
        img = self.preprocess(img)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        top_indices = pred[0].argsort()[-3:][::-1]

        output = "\n".join([f"{alpha[i]}: {pred[0][i]*100:.2f}%" for i in top_indices])
        self.label.config(text=f"Prediction:\n{output}")

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

if __name__ == "__main__":
    app = SketchPad()
    app.mainloop()
