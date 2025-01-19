import numpy as np
import sys
import subprocess

# Programming Assignment 4 tests
# Meant to be run on the Hydra cluster
# You will lose points if your code does not pass these tests

proc = subprocess.Popen(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = proc.communicate()

# Decode the output if needed
stdout = stdout.decode("utf-8")
stderr = stderr.decode("utf-8")
if proc.returncode != 0:
  print("Build failed")
  print(stderr)
  sys.exit()

shared_dir = '/mnt/coe/workspace/ece/ece786-spr24/conv2d/'

test_inputs = ["conv_input0.txt",
               "conv_input1.txt",
               "conv_input2.txt",
               "conv_input3.txt",
               "conv_input4.txt",
               "conv_input5.txt"]

test_filters = ["conv_filter0.txt",
               "conv_filter1.txt",
               "conv_filter2.txt",
               "conv_filter3.txt",
               "conv_filter4.txt",
               "conv_filter5.txt"]

expected_outputs = ["conv_output0.txt",
                    "conv_output1.txt",
                    "conv_output2.txt",
                    "conv_output3.txt",
                    "conv_output4.txt",
                    "conv_output5.txt"]

for idx,input in enumerate(test_inputs):
  print(f"Running Test{idx}...")
  proc = subprocess.Popen(["./conv2dV2", shared_dir+test_inputs[idx], shared_dir+test_filters[idx]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = proc.communicate()

  # Decode the output if needed
  stdout = stdout.decode("utf-8")
  stderr = stderr.decode("utf-8")

  if proc.returncode != 0:
    print(f"Test{idx} failed to run!")
    if stderr:
      print("Error message from subprocess:")
      print(stderr)
    sys.exit()

  stdout_output = stdout
  expected_output = np.genfromtxt(shared_dir+expected_outputs[idx])
  # Split the stdout into lines
  lines = stdout_output.strip().split('\n')
  # Convert the lines to a 2D list
  cuda_output = np.array([[float(val) for val in line.split()] for line in lines])

  if cuda_output.shape != expected_output.shape:
    print("Shapes don't match!!")
    sys.exit()

  diff = np.abs(cuda_output-expected_output)

  if np.all(diff <= 0.002):
    print(f"Test{idx} passed.")
  else:
    print(f"Test{idx} failed.")
    sys.exit()
