import csv
import os
import sys
import webview
import time
import threading


class BirdiaryViewer:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.window = None

    def start(self):
        class API:
            def next(api_self):
                if self.index < len(self.data) - 1:
                    self.index += 1
                    print(f"Loading URL: {self.data[self.index]['validation_link']}")
                    self.window.load_url(self.data[self.index]["validation_link"])
                    threading.Thread(target=self.wait_and_inject).start()

            def prev(api_self):
                if self.index > 0:
                    self.index -= 1
                    print(f"Loading URL: {self.data[self.index]['validation_link']}")
                    self.window.load_url(self.data[self.index]["validation_link"])
                    threading.Thread(target=self.wait_and_inject).start()

            def goto(api_self, index):
                if 0 <= index < len(self.data):
                    self.index = index
                    self.window.load_url(self.data[self.index]["validation_link"])
                    threading.Thread(target=self.wait_and_inject).start()

        self.window = webview.create_window(
            "Birdiary Viewer",
            url=self.data[self.index]["validation_link"],
            width=1920,
            height=1080,
            js_api=API(),
        )

        webview.start(self.add_controls)

    def wait_and_inject(self):
        time.sleep(1.5)  # Wait for page to load (adjust if needed)
        self.add_controls()

    # Run JavaScript to inject navigation buttons into the page
    def add_controls(self):
        js = f"""
        const existingBar = document.getElementById('birdiary-bar');
        if (existingBar) existingBar.remove();

        const navBar = document.createElement('div');
        navBar.id = 'birdiary-bar';
        navBar.style.position = 'fixed';
        navBar.style.top = '0';
        navBar.style.width = '100%';
        navBar.style.background = '#ddec6d';
        navBar.style.padding = '12px 20px';
        navBar.style.fontFamily = 'arial';
        navBar.style.zIndex = '9999';
        navBar.style.boxSizing = 'border-box';
        navBar.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.2)';

        navBar.innerHTML = `
            <div style="margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                <input id="birdiary-url" type="text" readonly style="
                    width: 100%;
                    padding: 6px 10px;
                    font-size: 14px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: #f9f9f9;">
                <button onclick="navigator.clipboard.writeText(document.getElementById('birdiary-url').value)" style="
                    font-size: 14px;
                    padding: 6px 10px;
                    border: none;
                    background-color: #58bb2a;
                    color: white;
                    border-radius: 4px;
                    cursor: pointer;">
                    Copy
                </button>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 16px; margin-bottom: 10px;">
                <div style="font-weight: 500;">Thank you for helping me verify the data!</div>
                <div style="font-style: italic;">Birdiary WebViewer to validate a list of observations</br>made by Anni Kurkela for my Master Thesis</div>
            </div>
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap;">
                <button onclick="pywebview.api.prev()" style="
                    padding: 10px 20px;
                    font-size: 16px;
                    border: none;
                    border-radius: 6px;
                    background-color: #58bb2a;
                    color: white;
                    cursor: pointer;">
                    ← Previous
                </button>
                <div id="birdiary-index" style="font-size: 16px; font-weight: bold;"></div>
                <button onclick="pywebview.api.next()" style="
                    padding: 10px 20px;
                    font-size: 16px;
                    border: none;
                    border-radius: 6px;
                    background-color: #58bb2a;
                    color: white;
                    cursor: pointer;">
                    Next →
                </button>
                <input id="goto-input" type="number" min="1" max="{len(self.data)}" placeholder="#" style="
                    width: 60px;
                    padding: 6px 6px;
                    font-size: 14px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                    text-align: center;">
                <button onclick="const idx = parseInt(document.getElementById('goto-input').value); 
                                if (!isNaN(idx) && idx >= 1 && idx <= {len(self.data)}) {{
                                    pywebview.api.goto(idx - 1);
                                }}" style="
                    font-size: 14px;
                    padding: 6px 12px;
                    border: none;
                    background-color: #58bb2a;
                    color: white;
                    border-radius: 4px;
                    cursor: pointer;">
                    Go
                </button>
            </div>
        `;
        document.body.appendChild(navBar);
        document.body.style.paddingTop = '200px';
        """

        self.window.evaluate_js(js)

        current_url = self.data[self.index]["validation_link"]
        index_display = f"{self.index + 1} / {len(self.data)}"

        self.window.evaluate_js(f'document.getElementById("birdiary-index").innerText = "{index_display}";')
        self.window.evaluate_js(f'document.getElementById("birdiary-url").value = "{current_url}";')




def load_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [
            {"validation_link": row["validation_link"]}
            for row in reader
            if row["validation_link"]
        ]


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_file = os.path.join(base_dir, "birdiary_webview_data.csv")

    data = load_csv(csv_file)
    if not data:
        raise SystemExit("No valid data found in CSV.")

    app = BirdiaryViewer(data)
    app.start()
