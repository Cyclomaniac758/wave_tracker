# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:55:35 2021

@author: Icarus
"""
import os

# We are going to dump our output in a folder in the same directory.
OUTPUT_DIR = "output"

# Names of output files to be written to output/ go here:
RECOGNIZED_WAVE_REPORT_FILE = "recognized_waves.txt"
CAM_SPEED = 30

def write_report(waves, performance):
    """Takes a list of recognized wave objects and writes attributes
    out to a plain text file.  Supplements this information with
    program performance and user stats.

    Args:
      waves: a list of wave objects
      performance_metric: a double representing speed of program

    Returns:
      NONE: writes the report to a txt file.
    """
    # Provide User feedback here.
    if not waves:
        print("No waves found.  No report written.")
    else:
        print("Writing analysis report to", RECOGNIZED_WAVE_REPORT_FILE)

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Write recognized waves to text file.
    report_path = os.path.join(OUTPUT_DIR, RECOGNIZED_WAVE_REPORT_FILE)

    # Use the context manager here:
    with open(report_path, "w") as text_file:
        text_file.write("Program performance: {} frames per second.\n"
                        .format(performance))
        for i, wave in enumerate(waves):
            text_file.write("Wave #{}: ID: {}, Wave Face Duration: {}s\n"
                            .format(i+1, wave.name, (wave.death - wave.birth)/CAM_SPEED))
            
