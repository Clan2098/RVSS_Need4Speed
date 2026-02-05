from pynput import keyboard

from steer_labels import LABELS, LABEL_TO_ANGLE


class Teleop:
    def __init__(self, on_error=None, on_toggle_stop=None):
        self.angle = 0
        self.label = 3
        self.write_images = False
        self.continue_running = True
        self._on_error = on_error
        self._on_toggle_stop = on_toggle_stop
        self._listener = keyboard.Listener(on_press=self._on_press)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.label = 3
            elif key == keyboard.Key.down:
                self.label = 3
            elif key == keyboard.Key.right:
                self.label = min(self.label + 1, len(LABELS) - 1)
            elif key == keyboard.Key.left:
                self.label = max(self.label - 1, 0)
            elif key == keyboard.Key.space:
                print("Toggle stop")
                if self._on_toggle_stop is not None:
                    self._on_toggle_stop()
            elif key == keyboard.Key.esc:
                print("Stopping script")
                self.continue_running = False
            elif key.char == 'c':
                print("Toggle write images")
                self.write_images = not self.write_images
                if self.write_images:
                    print("Writing images to folder")
                else:
                    print("Not writing images to folder")

            self.angle = LABEL_TO_ANGLE[self.label]

        except Exception as e:
            print(f"An error occurred: {e}")
            if self._on_error is not None:
                self._on_error(e)
