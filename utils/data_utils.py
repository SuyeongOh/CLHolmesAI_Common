def rhythmLabelEpisodeFinder(target_label: str, record: str, rhythm_label_pid, all_frame_annotation):
    start_afib_flag = False
    start_afib_idx = 0
    record_label_info = []
    for i, label in rhythm_label_pid:
        if target_label in label:
            start_afib_flag = True
            start_afib_idx = i
            continue
        if start_afib_flag:
            try:
                label_info = {}
                label_info['start_idx'] = start_afib_idx
                label_info['start_sample'] = all_frame_annotation[record][start_afib_idx]
                label_info['end_idx'] = i
                label_info['end_sample'] = all_frame_annotation[record][i]
                record_label_info.append(label_info)
                start_afib_flag = False
            except Exception as e:
                continue

    if start_afib_flag:
        label_info = {}
        label_info['start_idx'] = start_afib_idx
        label_info['start_sample'] = all_frame_annotation[record][start_afib_idx]
        label_info['end_idx'] = -1
        label_info['end_sample'] = 650000
        record_label_info.append(label_info)
    return record_label_info