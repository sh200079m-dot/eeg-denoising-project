import os
import mne

def load_eeg(subject_id):
    """
    Load EEG data for a given subject from Windows path
    """
    # مسیر دیتاست کامل در ویندوز (بعد از استخراج)
    win_path = "/mnt/c/Users/Asus/Downloads/dataset_Rest eyes open - Parkinsons Disease 64-Channel EEG/ds004584-download"
    
    # ساخت مسیر برای سوژه مورد نظر
    subject_dir = os.path.join(win_path, subject_id, "eeg")
    
    print(f"Looking for subject data in: {subject_dir}")
    
    # بررسی وجود پوشه
    if not os.path.exists(subject_dir):
        # بعضی دیتاست‌ها ممکنه مستقیماً توی پوشه سوژه باشن
        subject_dir = os.path.join(win_path, subject_id)
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"❌ Path not found for {subject_id}: {subject_dir}")
    
    # پیدا کردن فایل .set
    set_files = [f for f in os.listdir(subject_dir) if f.endswith('.set')]
    
    if not set_files:
        raise FileNotFoundError(f"❌ No .set file found in {subject_dir}")
    
    eeg_file = os.path.join(subject_dir, set_files[0])
    print(f"✅ Loading EEG file: {eeg_file}")
    
    # بارگذاری داده
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    return raw