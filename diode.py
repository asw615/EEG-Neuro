import numpy as np
import mne

# Step 1: Load the data
raw = mne.io.read_raw_fif("data/group9_own_AH_ver3-raw.fif", preload=True)
raw.info['bads'] = []
print("Channel names:", raw.info['ch_names'])

# Step 2: Set montage and reference
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, verbose=False)

# Select only EEG and stimulus channels
raw.pick_types(meg=False, eeg=True, eog=True, misc=True, exclude=[])
raw.set_eeg_reference(ref_channels='average', ch_type='eeg')

# Step 3: Locate and clean stimulus events
events, event_id = mne.events_from_annotations(raw)

# Remove spurious triggers (e.g., if dif < 2 ms)
dif_onsets = np.diff(events[:, 0])
dif_idx = np.where(dif_onsets < 2)[0]
events = np.delete(events, dif_idx, axis=0)
print(f"Events with index {dif_idx} were rejected.")

# Print unique event IDs after cleaning
print("Unique event IDs after cleaning:", np.unique(events[:, 2]))

# Step 4: Rename and filter relevant events for diode check
diode_event_id = {
    'Prime_M': 11,  # prime masculine
    'Prime_F': 12,  # prime feminine
    'Prime_N': 13  # prime neutral
#    'Target_M': 21, # target masculine
#    'Target_F': 22, # target feminine
#    'Target_N': 23  # target neutral
}

# Verify and update diode event IDs
updated_diode_event_id = {k: v for k, v in diode_event_id.items() if v in np.unique(events[:, 2])}
print("Updated diode event IDs:", updated_diode_event_id)

# Check if updated_diode_event_id is not empty
if not updated_diode_event_id:
    print("No matching events found for the specified diode event IDs.")
else:
    # Step 5: Create epochs for diode checking
    tmin_d, tmax_d = -0.01, 0.05  # Narrow window for inspecting delays
    baseline = None  # No baseline correction here to manually adjust later

    epochs_diode = mne.Epochs(
        raw,
        events=events,
        event_id=updated_diode_event_id,
        tmin=tmin_d,
        tmax=tmax_d,
        baseline=baseline,  # No baseline correction
        preload=True,  # Preload the data to allow direct manipulation
        verbose=False,
    )

    # Step 6: Adjust baseline to make minimum value zero
    diode_channel_index = epochs_diode.ch_names.index('41')
    data = epochs_diode.get_data(picks=diode_channel_index)
    min_value = data.min()
    adjusted_data = data - min_value

    # Create a new info object for the diode channel
    diode_info = mne.create_info(
        ch_names=['41'],
        sfreq=raw.info['sfreq'],
        ch_types=['misc']  # Assuming '41' is a miscellaneous channel
    )

    # Create a new epochs object with adjusted data
    adjusted_epochs_diode = mne.EpochsArray(
        adjusted_data,
        info=diode_info,
        events=epochs_diode.events,
        tmin=epochs_diode.tmin,
        event_id=epochs_diode.event_id,
    )

    # Plot adjusted diode epochs
    adjusted_epochs_diode.plot_image(picks='41')
