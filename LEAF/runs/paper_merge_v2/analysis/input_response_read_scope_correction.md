# Correction to the input-response diagnostic's EEG read scope

Inspection of the frozen `diagnose_input_responses_v3.py` and
`diagnose_sensor_subspaces.py:row_map` establishes a narrower claim than one
sentence in the original design and applicability note.

For subject-mapped tasks, `row_map(..., 'calibration')` selects the existing
training rows by the modulo5 split. However, the input-response worker first
copies entire selected subject datasets with `h[key][:]` into host memory,
then extracts the selected training rows from those copies. Consequently it
also reads the unused validation EEG rows stored in those same subject
datasets. The statement that it performed no validation EEG reads was too
strong. The predefined-split tasks read their trainX datasets.

The actual diagnostic tensors are checked against exact training EEG hashes,
training IDs and labels. Predictions and Jacobians use only the selected
training rows. No validation inference, optimization or official-test read
occurs in that diagnostic. Its numerical results and frozen source files are
preserved; this corrects the description of I/O, not the evaluated samples.

The new `cache_preprojection_representations.py` requests only selected EEG
row indices directly from HDF5, with duplicate indices removed before the
read and restored afterward. Partition labels are still read to reconstruct
the existing split. It replays the input-response subset EEG and label hashes
before any model inference. Its frozen design records this correction.
