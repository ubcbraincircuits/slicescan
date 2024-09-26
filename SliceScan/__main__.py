import os
import tkinter as tk
from tkinter import filedialog, ttk
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
from matplotlib.widgets import RectangleSelector
import pandas as pd
import cv2
import time
# Initialize empty lists and variables
hi_values = []
lo_values = []
trial_num = []
mean_values = []
drug_status = []
roi_coords = None  # Store ROI coordinates
electrode_coords = None  # Store electrode coordinates
drug_status = {'Drug 1': None, 'Drug 2': None}
experiment_running = True
im_display = None
drug_ranges = {}
line = None
start_point = None
end_point = None
drugs_in_legend = []
legend_proxies = []
existing_legend_handles = []
existing_legend_labels = []
drug1_periods = []
drug2_periods = []

def on_press(event):
    global start_point, line
    if event.button == 3:
        start_point = [event.xdata, event.ydata]
        end_point = start_point
        line, = ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r-')

def on_release(event):
    global end_point
    if event.button == 3:
        end_point = [event.xdata, event.ydata]
        line.set_data([start_point[0], end_point[0]], [start_point[1], end_point[1]])
        fig.canvas.draw()


# Add electrode line
def electrode_position_draw(eclick):
    global electrode_coords
    electrode_coords = (eclick.xdata, eclick.ydata)
    print(electrode_coords)


def line_select_callback(eclick, erelease):
    global roi_coords
    roi_coords = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
    print(f"ROI coordinates: {roi_coords}")
    #ax.add_patch(plt.Rectangle((eclick.xdata, eclick.ydata), erelease.xdata - eclick.xdata, erelease.ydata - eclick.ydata, fill=False, edgecolor='red', linewidth=2))
    canvas.draw()

# Draggable line for electrode
def onpick(event):
    global electrode_coords
    electrode_coords = event.artist.get_data()
    
# Add electrode line
#def add_electrode_line(ax):
#    line, = ax.plot([0, 50], [50, 100], picker=True, color='red')  # Initial dummy line
#    fig.canvas.mpl_connect('pick_event', onpick)

def read_tif_file(file_path):
    # Read tif file using tifffile library
    with tifffile.TiffFile(file_path) as tif:
        images = tif.asarray()
    # Convert 12-bit to 8-bit
    images_8bit = np.uint8(images / 10)
    return images_8bit

def calculate_dff(image_stack, baseline, roi_coords):
    x1, y1, x2, y2 = map(int, roi_coords)
    roi = image_stack[:, y1:y2, x1:x2]
    avg_intensity = np.mean(roi, axis=(1, 2))
    no_baseline = baseline[:, y1:y2, x1:x2]
    avg_baseline = np.mean(no_baseline, axis=(1, 2))
    dff = (avg_intensity - avg_baseline) / avg_baseline
    dff_stack = (image_stack - no_baseline) / no_baseline
    return dff, dff_stack

def toggle_drug(drug_name, check_var):
    global drug_ranges
    if not drug_name.get():  # Check if name is provided for the drug
        tk.messagebox.showerror("Error", "First put a name for the drug.")
        check_var.set(0)  # Reset the checkbox
        return
    drug = drug_name.get()
    if drug not in drug_ranges:
        drug_ranges[drug] = []
    if bool(check_var.get()):
        drug_ranges[drug].append((trial_num[-1], None))  # Start range
    else:
        start, _ = drug_ranges[drug][-1]
        drug_ranges[drug][-1] = (start, trial_num[-1])  # End range
    
    

def export_data():
    folder = tk.filedialog.askdirectory(title="Choose export directory")
    if not folder:
        return

    metadata = {
        'Date and Time': pd.Timestamp.now(),
        'Name of Experiment': experiment_name.get(),
        'Notes': notes.get(),
        'Name of Drug 1': drug1_name.get(),
        'Name of Drug 2': drug2_name.get(),
        'ROI Coordinates': str(roi_coords),
        'Electrode Coordinates': str(electrode_coords)
    }

    df = pd.DataFrame({
        'Trial': trial_num,
        'Area': mean_values,
        'Volume': hi_values,
        'Wash': ['aCSF'] * len(trial_num)  # Placeholder for drug wash status
    })

    wash_column = []
    for i, trial in enumerate(trial_num):
        wash = []
        for drug, periods in drug_ranges.items():
            for start, end in periods:
                if start <= trial <= (end if end is not None else trial):
                    wash.append(drug)
        wash_str = ', '.join(wash) if wash else 'aCSF'
        wash_column.append(wash_str)

    df['Wash'] = wash_column

    with pd.ExcelWriter(os.path.join(folder, 'Experiment_Data.xlsx')) as writer:
        pd.DataFrame(metadata, index=[0]).to_excel(writer, sheet_name='Metadata')
        df.to_excel(writer, sheet_name='Data')
    # Save the first and last images
    first_image = read_tif_file(os.path.join(folder_path.get(), "hi1.tif"))[1]
    last_image = read_tif_file(os.path.join(folder_path.get(), f"hi{trial_num[-1]}.tif"))[1]
    
    for idx, image in enumerate([first_image, last_image]):
        min_val = np.min(image)
        max_val = np.max(image)
        image_8bit = np.uint8((image - min_val) / (max_val - min_val)*255)

        # Draw rectangle
        cv2.rectangle(image_8bit, (int(roi_coords[0]), int(roi_coords[1])), (int(roi_coords[2]), int(roi_coords[3])), (255, 0, 0), 2)

        # Save image
        cv2.imwrite(os.path.join(folder, f'image_{idx + 1}.tif'), image_8bit)
        fig2.savefig(os.path.join(folder, 'Plot.png'))
        
def update_displayed_image(image, roi_coords, electrode_coords):
    global im_display
    if im_display is None:
        im_display = ax.imshow(image, cmap='gray')  # Initialize the plot
        if roi_coords:
            ax.add_patch(plt.Rectangle((roi_coords[0], roi_coords[1]), roi_coords[2] - roi_coords[0], roi_coords[3] - roi_coords[1], edgecolor='red', facecolor='none'))
        if electrode_coords:
            ax.plot(electrode_coords[0], electrode_coords[1], color='blue')
    else:
        im_display.set_array(image)  # Update the plot
    canvas.draw()

def browse_folder():
    global roi_coords, electrode_coords
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)
    
    # Assuming you've loaded the first image already
    first_image_path = os.path.join(folder_selected, "hi1.tif")
    if os.path.exists(first_image_path):
        first_image = read_tif_file(first_image_path)[0]
        
        ax.imshow(first_image, cmap='gray')  #  'ax' is existing matplotlib axis
        
        # Enable Rectangle Selector for ROI
        if roi_coords==None:
            rectangle_selector = RectangleSelector(ax, line_select_callback,
                                            useblit=True, # drawtype='box',
                                            button=[1],  # don't use middle button
                                            minspanx=5, minspany=5,
                                            spancoords='pixels',
                                            interactive=True)
        
            # Enable line selector for electrode
            fig.canvas.mpl_connect('button_release_event', rectangle_selector)
        if electrode_coords==None:
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('button_release_event', on_release)
        canvas.draw()
        print(electrode_coords)
def start_experiment():
    global experiment_running
    experiment_running = True

    

    def check_file_existence():
        global roi_coords, electrode_coords, experiment_running, existing_legend_handles, existing_legend_labels, drugs_in_legend
        folder = folder_path.get()
        counter = 1  # To keep track of the trial number
        while experiment_running:
            hi_path = os.path.join(folder, f"hi{counter}.tif")
            lo_path = os.path.join(folder, f"lo{counter}.tif")
            no_path = os.path.join(folder, f"no{counter}.tif")
            
            if all(map(os.path.exists, [hi_path, no_path])):
                time.sleep(2)
                hi_images = read_tif_file(hi_path)
                lo_images = read_tif_file(lo_path)
                no_images = read_tif_file(no_path)
                # Update the first frame image display
                time.sleep(2)
                first_frame = hi_images[0]
                update_displayed_image(first_frame, roi_coords, electrode_coords)
                hi_dff, hi_stack_dff = calculate_dff(hi_images, no_images, roi_coords)
                lo_dff, lo_stack_dff = calculate_dff(lo_images, no_images, roi_coords)
                avg_stack_dff = (hi_stack_dff + lo_stack_dff)/2
                
                hi_max = np.max(hi_dff)
                lo_max = np.max(lo_dff)
                
                # Assume avg_stack_dff is your n x 128 x 128 ndarray
                n, height, width = avg_stack_dff.shape
                
                # Calculate the standard deviation over time for each pixel
                std_dev = np.std(avg_stack_dff, axis=0)
                
                # Set the threshold as 4 times the standard deviation
                threshold = 4 * std_dev
                
                # Generate the mask stack
                mask_stack = avg_stack_dff > threshold
                
                # Calculate the sum of all the masks (total number of activated pixels)
                total_activated_pixels = np.sum(mask_stack)
                
                # Calculate the maximum mask value over time for each pixel
                max_mask_over_time = np.max(mask_stack, axis=0)
                
                max_area = np.sum(max_mask_over_time)
                
                # Now, total_activated_pixels holds the sum of all activated pixels across all frames
                # and max_mask_over_time holds the maximum activation state (0 or 1) of each pixel over time
                    
                
                # Update lists and plot
                hi_values.append(total_activated_pixels)
                #lo_values.append(lo_max)
                trial_num.append(counter)
                mean_values.append((max_area))
                
                ax2.clear()
                ax2.plot(trial_num, total_activated_pixels, label='Volume', linestyle='--', alpha=0.5)
                #ax2.plot(trial_num, lo_values, label='LO', linestyle='--', alpha=0.5)
                ax2.plot(trial_num, max_area, label='Area', color='black')
                

                # Add drug wash color fill
                color_map = {drug1_name.get(): 'blue', drug2_name.get(): 'green'}
                existing_legend_handles, existing_legend_labels = ax2.get_legend_handles_labels()


                for drug, ranges in drug_ranges.items():
                    for start, end in ranges:
                        if end is None:
                            end = trial_num[-1]  # Use the current trial as the end if the range is not complete
                        ax2.axvspan(start, end, alpha=0.25, color=color_map.get(drug, 'red'),edgecolor='none')

                    if drug not in existing_legend_labels:
                        proxy = matplotlib.patches.Patch(color=color_map.get(drug, 'red'), alpha=0.25)
                        
                        existing_legend_handles.append(proxy)
                        existing_legend_labels.append(drug)
                        #legend_proxies.append((proxy, drug))
                        drugs_in_legend.append(drug)

               
                #ax2.legend()
                #ax2.legend([x[0] for x in legend_proxies], [x[1] for x in legend_proxies], loc='upper right')
                ax2.legend(existing_legend_handles, existing_legend_labels, loc='upper right')
                canvas2.draw()
                
                counter += 1  # Increment the trial number
            else:
                time.sleep(1)
    # Use a thread to continuously check for file existence
    thread = threading.Thread(target=check_file_existence)
    thread.start()

def stop_experiment():
    global experiment_running
    answer = tk.messagebox.askyesno("Confirmation", "Do you really want to stop the experiment?")
    if answer:
        experiment_running = False



# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Brain Slice Experiment")
# Make the grid responsive
for i in range(3):
    root.columnconfigure(i, weight=1)
for i in range(5, 9):
    root.rowconfigure(i, weight=1)
root.rowconfigure(6, weight=0)

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(True)
fig2, ax2 = plt.subplots()
# For square fig, aspect ratio is 1:1

canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.get_tk_widget().grid(row=5, columnspan=3)
canvas2 = FigureCanvasTkAgg(fig2, master=root)
#canvas2.get_tk_widget().grid(row=8, columnspan=3)

canvas.get_tk_widget().grid(row=5, column=1)
canvas.get_tk_widget().config(width=300, height=300)
canvas2.get_tk_widget().grid(row=8, columnspan=3, sticky="nsew")

# Folder path input
folder_path = tk.StringVar()
tk.Label(root, text="Folder Path:").grid(row=0, column=0)
tk.Entry(root, textvariable=folder_path).grid(row=0, column=1)
#tk.Button(root, text="Browse", command=lambda: folder_path.set(filedialog.askdirectory())).grid(row=0, column=2)
tk.Button(root, text="Browse", command=browse_folder).grid(row=0, column=2)

# Experiment name and notes
experiment_name = tk.StringVar()
notes = tk.StringVar()
tk.Label(root, text="Experiment Name:").grid(row=1, column=0)
tk.Entry(root, textvariable=experiment_name).grid(row=1, column=1)
tk.Label(root, text="Notes:").grid(row=2, column=0)
tk.Entry(root, textvariable=notes).grid(row=2, column=1)

# Drug 1
drug1_name = tk.StringVar()
drug1_check = tk.IntVar()
tk.Label(root, text="Drug 1:").grid(row=3, column=0)
tk.Entry(root, textvariable=drug1_name).grid(row=3, column=1)
tk.Checkbutton(root, variable=drug1_check, command=lambda: toggle_drug(drug1_name, drug1_check)).grid(row=3, column=2)

# Drug 2
drug2_name = tk.StringVar()
drug2_check = tk.IntVar()
tk.Label(root, text="Drug 2:").grid(row=4, column=0)
tk.Entry(root, textvariable=drug2_name).grid(row=4, column=1)
tk.Checkbutton(root, variable=drug2_check, command=lambda: toggle_drug(drug2_name, drug2_check)).grid(row=4, column=2)

# Plotting

#canvas.mpl_connect("key_press_event", on_key_press)
# Buttons
tk.Button(root, text="Start Experiment", command=start_experiment).grid(row=6, column=0)
tk.Button(root, text="Stop Experiment", command=stop_experiment).grid(row=6, column=1)
tk.Button(root, text="Export Data", command=export_data).grid(row=6, column=2)

# Run the GUI
root.mainloop()
