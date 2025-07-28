# không xuống hàng

# import pytesseract
# from PIL import Image, ImageTk
# from tkinter import Tk, filedialog, Canvas, Button
# import numpy as np
#
#
# def Img_to_text():
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
#     class ImageCropApp:
#         def __init__(self, root):
#             self.root = root
#             self.root.title("Crop Image and Extract Text")
#
#             self.canvas = Canvas(root, width=800, height=600)
#             self.canvas.pack(fill="both", expand=True)
#
#             self.btn_load = Button(root, text="Load Image", command=self.load_image)
#             self.btn_load.pack()
#
#             self.btn_crop = Button(root, text="Crop and Extract Text", command=self.crop_and_extract_text)
#             self.btn_crop.pack()
#
#             self.image = None
#             self.tk_image = None
#             self.start_x = None
#             self.start_y = None
#             self.rect_id = None
#             self.margin = 20
#             self.extracted_text = None  # To store the extracted text
#
#             self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#             self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
#             self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
#
#         def load_image(self):
#             file_path = filedialog.askopenfilename(title="Select an Image",
#                                                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
#             if file_path:
#                 self.image = Image.open(file_path)
#                 self.update_canvas()
#
#         def update_canvas(self):
#             if self.image:
#                 self.canvas.config(width=self.image.width + 2 * self.margin, height=self.image.height + 2 * self.margin)
#                 self.tk_image = ImageTk.PhotoImage(self.image)
#                 self.canvas.delete("all")
#                 self.canvas.create_image(self.margin, self.margin, anchor="nw", image=self.tk_image)
#
#         def on_button_press(self, event):
#             self.start_x = self.canvas.canvasx(event.x)
#             self.start_y = self.canvas.canvasy(event.y)
#
#             if self.rect_id:
#                 self.canvas.delete(self.rect_id)
#
#             self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
#                                                         outline="red")
#
#         def on_mouse_drag(self, event):
#             cur_x = self.canvas.canvasx(event.x)
#             cur_y = self.canvas.canvasy(event.y)
#             self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
#
#         def on_button_release(self, event):
#             pass
#
#         def crop_and_extract_text(self):
#             if self.start_x and self.start_y and self.rect_id:
#                 end_x, end_y = self.canvas.coords(self.rect_id)[2:]
#                 x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
#                 x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))
#
#                 x1 = max(0, x1 - self.margin)
#                 y1 = max(0, y1 - self.margin)
#                 x2 = min(self.image.width, x2 + self.margin)
#                 y2 = min(self.image.height, y2 + self.margin)
#
#                 img_cv = np.array(self.image)
#                 cropped_img_cv = img_cv[y1:y2, x1:x2]
#
#                 cropped_img_pil = Image.fromarray(cropped_img_cv)
#                 text = pytesseract.image_to_string(cropped_img_pil)
#
#                 lines = text.split('\n')
#                 non_empty_lines = [line for line in lines if line.strip() != '']
#
#                 # Join lines without additional newlines
#                 self.extracted_text = ' '.join(non_empty_lines)  # Use a space to join the lines
#
#                 print(self.extracted_text)
#                 self.root.quit()
#             else:
#                 print("No cropping area selected.")
#                 self.extracted_text = None
#
#         def get_extracted_text(self):
#             return self.extracted_text
#
#     root = Tk()
#     app = ImageCropApp(root)
#     root.mainloop()
#
#     return app.get_extracted_text()
#
#
# # Example usage
# extracted_text = Img_to_text()
# print("Extracted Text:", extracted_text)

# xuống hàng

"""import pytesseract
from PIL import Image, ImageTk
from tkinter import Tk, filedialog, Canvas, Button
import numpy as np


def Img_to_text():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    class ImageCropApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Crop Image and Extract Text")

            self.canvas = Canvas(root, width=800, height=600)
            self.canvas.pack(fill="both", expand=True)

            self.btn_load = Button(root, text="Load Image", command=self.load_image)
            self.btn_load.pack()

            self.btn_crop = Button(root, text="Crop and Extract Text", command=self.crop_and_extract_text)
            self.btn_crop.pack()

            self.image = None
            self.tk_image = None
            self.start_x = None
            self.start_y = None
            self.rect_id = None
            self.margin = 20
            self.extracted_text = None  # To store the extracted text

            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        def load_image(self):
            file_path = filedialog.askopenfilename(title="Select an Image",
                                                   filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if file_path:
                self.image = Image.open(file_path)
                self.update_canvas()

        def update_canvas(self):
            if self.image:
                self.canvas.config(width=self.image.width + 2 * self.margin, height=self.image.height + 2 * self.margin)
                self.tk_image = ImageTk.PhotoImage(self.image)
                self.canvas.delete("all")
                self.canvas.create_image(self.margin, self.margin, anchor="nw", image=self.tk_image)

        def on_button_press(self, event):
            self.start_x = self.canvas.canvasx(event.x)
            self.start_y = self.canvas.canvasy(event.y)

            if self.rect_id:
                self.canvas.delete(self.rect_id)

            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                        outline="red")

        def on_mouse_drag(self, event):
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

        def on_button_release(self, event):
            pass

        def crop_and_extract_text(self):
            if self.start_x and self.start_y and self.rect_id:
                end_x, end_y = self.canvas.coords(self.rect_id)[2:]
                x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
                x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))

                x1 = max(0, x1 - self.margin)
                y1 = max(0, y1 - self.margin)
                x2 = min(self.image.width, x2 + self.margin)
                y2 = min(self.image.height, y2 + self.margin)

                img_cv = np.array(self.image)
                cropped_img_cv = img_cv[y1:y2, x1:x2]

                cropped_img_pil = Image.fromarray(cropped_img_cv)
                text = pytesseract.image_to_string(cropped_img_pil)

                lines = text.split('\n')
                final_text = []
                empty_line_count = 0

                for line in lines:
                    if line.strip() == '':
                        empty_line_count += 1
                    else:
                        if empty_line_count > 0:
                            final_text.append('')  # Add a line break
                        final_text.append(line.strip())
                        empty_line_count = 0

                # Join lines with a single space for continuous text
                self.extracted_text = '\n'.join(final_text)

                print(self.extracted_text)
                self.root.quit()
            else:
                print("No cropping area selected.")
                self.extracted_text = None

        def get_extracted_text(self):
            return self.extracted_text

    root = Tk()
    app = ImageCropApp(root)
    root.mainloop()

    return app.get_extracted_text()


# Example usage
extracted_text = Img_to_text()
print(extracted_text)
"""

# khi có chấm cuối câu xuống dòng
# import pytesseract
# from PIL import Image, ImageTk
# from tkinter import Tk, filedialog, Canvas, Button
# import numpy as np
#
#
# def Img_to_text():
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
#     class ImageCropApp:
#         def __init__(self, root):
#             self.root = root
#             self.root.title("Crop Image and Extract Text")
#
#             self.canvas = Canvas(root, width=800, height=600)
#             self.canvas.pack(fill="both", expand=True)
#
#             self.btn_load = Button(root, text="Load Image", command=self.load_image)
#             self.btn_load.pack()
#
#             self.btn_crop = Button(root, text="Crop and Extract Text", command=self.crop_and_extract_text)
#             self.btn_crop.pack()
#
#             self.image = None
#             self.tk_image = None
#             self.start_x = None
#             self.start_y = None
#             self.rect_id = None
#             self.margin = 20
#             self.extracted_text = None  # To store the extracted text
#
#             self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#             self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
#             self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
#
#         def load_image(self):
#             file_path = filedialog.askopenfilename(title="Select an Image",
#                                                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
#             if file_path:
#                 self.image = Image.open(file_path)
#                 self.update_canvas()
#
#         def update_canvas(self):
#             if self.image:
#                 self.canvas.config(width=self.image.width + 2 * self.margin, height=self.image.height + 2 * self.margin)
#                 self.tk_image = ImageTk.PhotoImage(self.image)
#                 self.canvas.delete("all")
#                 self.canvas.create_image(self.margin, self.margin, anchor="nw", image=self.tk_image)
#
#         def on_button_press(self, event):
#             self.start_x = self.canvas.canvasx(event.x)
#             self.start_y = self.canvas.canvasy(event.y)
#
#             if self.rect_id:
#                 self.canvas.delete(self.rect_id)
#
#             self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
#                                                         outline="red")
#
#         def on_mouse_drag(self, event):
#             cur_x = self.canvas.canvasx(event.x)
#             cur_y = self.canvas.canvasy(event.y)
#             self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
#
#         def on_button_release(self, event):
#             pass
#
#         def crop_and_extract_text(self):
#             if self.start_x and self.start_y and self.rect_id:
#                 end_x, end_y = self.canvas.coords(self.rect_id)[2:]
#                 x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
#                 x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))
#
#                 x1 = max(0, x1 - self.margin)
#                 y1 = max(0, y1 - self.margin)
#                 x2 = min(self.image.width, x2 + self.margin)
#                 y2 = min(self.image.height, y2 + self.margin)
#
#                 img_cv = np.array(self.image)
#                 cropped_img_cv = img_cv[y1:y2, x1:x2]
#
#                 cropped_img_pil = Image.fromarray(cropped_img_cv)
#                 text = pytesseract.image_to_string(cropped_img_pil)
#
#                 lines = text.split('\n')
#                 final_text = []
#
#                 for line in lines:
#                     stripped_line = line.strip()
#                     if stripped_line:  # Only process non-empty lines
#                         if final_text and final_text[-1].endswith('.'):
#                             final_text.append(stripped_line)  # New line after a period
#                         else:
#                             if final_text:
#                                 final_text[-1] += ' ' + stripped_line  # Join with a space
#                             else:
#                                 final_text.append(stripped_line)  # First line
#
#                 # Join the final text with newlines
#                 self.extracted_text = '\n'.join(final_text)
#
#                 print(self.extracted_text)
#                 self.root.quit()
#             else:
#                 print("No cropping area selected.")
#                 self.extracted_text = None
#
#         def get_extracted_text(self):
#             return self.extracted_text
#
#     root = Tk()
#     app = ImageCropApp(root)
#     root.mainloop()
#
#     return app.get_extracted_text()



"""import multiprocessing
import pytesseract
from PIL import Image, ImageTk
from tkinter import Tk, filedialog, Canvas, Button
import numpy as np

# Function to be run in a separate process
def run_tkinter(queue):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    class ImageCropApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Crop Image and Extract Text")

            self.canvas = Canvas(root, width=800, height=600)
            self.canvas.pack(fill="both", expand=True)

            self.btn_load = Button(root, text="Load Image", command=self.load_image)
            self.btn_load.pack()

            self.btn_crop = Button(root, text="Crop and Extract Text", command=self.crop_and_extract_text)
            self.btn_crop.pack()

            self.image = None
            self.tk_image = None
            self.start_x = None
            self.start_y = None
            self.rect_id = None
            self.margin = 20
            self.extracted_text = None  # To store the extracted text

            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        def load_image(self):
            file_path = filedialog.askopenfilename(title="Select an Image",
                                                   filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if file_path:
                self.image = Image.open(file_path)
                self.update_canvas()

        def update_canvas(self):
            if self.image:
                self.canvas.config(width=self.image.width + 2 * self.margin, height=self.image.height + 2 * self.margin)
                self.tk_image = ImageTk.PhotoImage(self.image)
                self.canvas.delete("all")
                self.canvas.create_image(self.margin, self.margin, anchor="nw", image=self.tk_image)

        def on_button_press(self, event):
            self.start_x = self.canvas.canvasx(event.x)
            self.start_y = self.canvas.canvasy(event.y)

            if self.rect_id:
                self.canvas.delete(self.rect_id)

            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                        outline="red")

        def on_mouse_drag(self, event):
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

        def on_button_release(self, event):
            pass

        def crop_and_extract_text(self):
            if self.start_x and self.start_y and self.rect_id:
                end_x, end_y = self.canvas.coords(self.rect_id)[2:]
                x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
                x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))

                x1 = max(0, x1 - self.margin)
                y1 = max(0, y1 - self.margin)
                x2 = min(self.image.width, x2 + self.margin)
                y2 = min(self.image.height, y2 + self.margin)

                img_cv = np.array(self.image)
                cropped_img_cv = img_cv[y1:y2, x1:x2]

                cropped_img_pil = Image.fromarray(cropped_img_cv)
                text = pytesseract.image_to_string(cropped_img_pil)

                lines = text.split('\n')
                final_text = []

                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line:  # Only process non-empty lines
                        if final_text and final_text[-1].endswith('.'):
                            final_text.append(stripped_line)  # New line after a period
                        else:
                            if final_text:
                                final_text[-1] += ' ' + stripped_line  # Join with a space
                            else:
                                final_text.append(stripped_line)  # First line

                # Join the final text with newlines
                self.extracted_text = '\n'.join(final_text)

                # Send the extracted text back to the main process
                queue.put(self.extracted_text)

                self.root.quit()
            else:
                print("No cropping area selected.")
                self.extracted_text = None

    root = Tk()
    app = ImageCropApp(root)
    root.mainloop()

# Wrapper function to call Tkinter in a separate process
def Img_to_text():
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_tkinter, args=(queue,))
    p.start()
    p.join()

    # Get the extracted text from the process
    return queue.get()

"""
import pytesseract
from PIL import Image, ImageTk
from tkinter import Tk, filedialog, Canvas, Button
import numpy as np

# Đặt đường dẫn đến Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class ImageCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Image and Extract Text")
        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)

        self.btn_load = Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack()

        self.btn_crop = Button(root, text="Crop and Extract Text", command=self.crop_and_extract_text)
        self.btn_crop.pack()

        self.image = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            self.update_canvas()

    def update_canvas(self):
        if self.image:
            tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
            self.canvas.image = tk_image

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="red")

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        pass

    def crop_and_extract_text(self):
        if self.start_x and self.start_y and self.rect_id:
            end_x, end_y = self.canvas.coords(self.rect_id)[2:]
            x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
            x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))

            img_cv = np.array(self.image)
            cropped_img_cv = img_cv[y1:y2, x1:x2]

            cropped_img_pil = Image.fromarray(cropped_img_cv)
            text = pytesseract.image_to_string(cropped_img_pil)
            print(text)  # In ra văn bản đã trích xuất

if __name__ == "__main__":
    root = Tk()
    app = ImageCropApp(root)
    root.mainloop()



# import cv2
# import pytesseract
# from PIL import Image
# import numpy as np
# from tkinter import filedialog, Tk
#
#
# def Img_to_text():
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
#     class ImageCropApp:
#         def __init__(self):
#             self.image = None
#             self.clone = None
#             self.rect_start = None
#             self.rect_end = None
#             self.cropping = False
#             self.extracted_text = None
#             self.margin = 20  # Margin for cropping
#             self.button_region = None
#             self.close_clicked = False  # Flag to indicate if the close button was clicked
#
#         def load_image(self):
#             # Use Tkinter filedialog to select image file from computer
#             root = Tk()
#             root.withdraw()  # Hide the Tkinter window
#             file_path = filedialog.askopenfilename(
#                 title="Select an Image",
#                 filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
#             )
#             root.destroy()  # Close the hidden root window
#
#             if file_path:
#                 self.image = cv2.imread(file_path)
#                 self.clone = self.image.copy()  # Clone for refreshing display
#                 if self.image is None:
#                     print("Error loading image.")
#                     return False
#                 # Extend image for the button display
#                 self.add_close_button_to_image()
#                 return True
#             else:
#                 print("No file selected.")
#                 return False
import threading
"""import cv2
import pytesseract
from PIL import Image
import numpy as np
from tkinter import filedialog, Tk

# Helper function to open the file dialog
def open_file_dialog():
    root = Tk()
    root.withdraw()  # Hide the Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    root.destroy()  # Close the hidden root window
    return file_path

def Img_to_text():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    class ImageCropApp:
        def __init__(self):
            self.image = None
            self.clone = None
            self.rect_start = None
            self.rect_end = None
            self.cropping = False
            self.extracted_text = None
            self.margin = 20  # Margin for cropping
            self.button_region = None

        def load_image(self):
            # Use the helper function to open the file dialog in the main thread
            file_path = open_file_dialog()

            if file_path:
                self.image = cv2.imread(file_path)
                self.clone = self.image.copy()  # Clone for refreshing display
                if self.image is None:
                    print("Error loading image.")
                    return False
                self.add_close_button_to_image()
                return True
            else:
                print("No file selected.")
                return False
        def add_close_button_to_image(self):
            # Add space for the button under the image
            button_height = 50
            self.clone = np.vstack([self.clone, np.zeros((button_height, self.image.shape[1], 3), dtype=np.uint8)])

            # Set button text and region
            button_color = (0, 255, 0)
            self.button_region = ((0, self.clone.shape[0] - button_height),
                                  (self.clone.shape[1], self.clone.shape[0]))

            # Draw the button as a rectangle
            cv2.rectangle(self.clone, self.button_region[0], self.button_region[1], button_color, -1)

            # Add button label
            button_label = "CLOSE"
            font_scale = 1
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(button_label, font, font_scale, font_thickness)[0]
            text_x = (self.clone.shape[1] - text_size[0]) // 2
            text_y = self.clone.shape[0] - (button_height // 2) + (text_size[1] // 2)
            cv2.putText(self.clone, button_label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        def show_image(self):
            if self.image is not None:
                cv2.namedWindow("Image")
                cv2.setMouseCallback("Image", self.click_and_crop_or_close)

                while True:
                    cv2.imshow("Image", self.clone)  # Show updated image clone

                    # Exit if window is closed or 'q' is pressed or close button is clicked
                    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1 or self.close_clicked:
                        break
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                cv2.destroyAllWindows()

        def click_and_crop_or_close(self, event, x, y, flags, param):
            # Check if the close button is clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.button_region[0][1] <= y <= self.button_region[1][1]:  # Inside button Y range
                    print("Close button clicked!")
                    self.close_clicked = True  # Set flag to close window
                    return

                # Start cropping if not clicking on the button
                self.rect_start = (x, y)
                self.cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.cropping:
                    # Refresh the clone of the image to keep drawing the rectangle clearly
                    self.clone = self.image.copy()
                    self.add_close_button_to_image()  # Redraw the button
                    cv2.rectangle(self.clone, self.rect_start, (x, y), (0, 255, 0), 2)  # Change thickness to 1
                    cv2.imshow("Image", self.clone)

            elif event == cv2.EVENT_LBUTTONUP:
                self.rect_end = (x, y)
                self.cropping = False
                self.clone = self.image.copy()  # Reset the clone to the original without the rectangle
                self.add_close_button_to_image()  # Redraw the button
                cv2.imshow("Image", self.clone)  # Display the original image without rectangle
                self.crop_and_extract_text()

        def crop_and_extract_text(self):
            if self.rect_start and self.rect_end:
                x1, y1 = min(self.rect_start[0], self.rect_end[0]), min(self.rect_start[1], self.rect_end[1])
                x2, y2 = max(self.rect_start[0], self.rect_end[0]), max(self.rect_start[1], self.rect_end[1])

                # Remove margin adjustments
                # x1 = max(0, x1 - self.margin)
                # y1 = max(0, y1 - self.margin)
                # x2 = min(self.image.shape[1], x2 + self.margin)
                # y2 = min(self.image.shape[0], y2 + self.margin)

                # Ensure coordinates are within the image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.image.shape[1], x2)
                y2 = min(self.image.shape[0], y2)

                cropped_img = self.image[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

                # Extract text using pytesseract
                text = pytesseract.image_to_string(cropped_img_pil)

                lines = text.split('\n')
                final_text = []

                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line:  # Only process non-empty lines
                        if final_text and final_text[-1].endswith('.'):
                            final_text.append(stripped_line)  # New line after a period
                        else:
                            if final_text:
                                final_text[-1] += ' ' + stripped_line  # Join with a space
                            else:
                                final_text.append(stripped_line)  # First line

                self.extracted_text = '\n'.join(final_text)
                print(self.extracted_text)

        def get_extracted_text(self):
            return self.extracted_text

    app = ImageCropApp()

    if app.load_image():
        app.show_image()

    return app.get_extracted_text()

"""
# Example usage:
# extracted_text = Img_to_text()
# print("Extracted text:", extracted_text)
