import soundata

def get_train_test_clips_n_labels():
    clips = load_dataset()
    label2id = create_label_map(clips)
    # split by folds
    test_clips, train_clips = split_clips(clips)

    return train_clips, test_clips, label2id


def split_clips(clips, test_fold=10):
    train_clips = [c for c in clips if c.fold not in [test_fold]]
    test_clips = [c for c in clips if c.fold == test_fold]
    return test_clips, train_clips


def load_dataset(name="urbansound8k"):
    dataset = soundata.initialize(name)
    clips = [dataset.clip(cid) for cid in dataset.clip_ids]
    return clips

def create_label_map(clips):
    classes = sorted({c.class_label for c in clips})
    return {label: i for i, label in enumerate(classes)}