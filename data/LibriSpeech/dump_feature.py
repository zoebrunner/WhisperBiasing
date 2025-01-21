import os
import json
import torch
import whisper

# Specify GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Save output on disk
base_output_path = "/disk/data4/zbrunner/whisper_biasing"

# Shared corpus storage
base_path = "/afs/inf.ed.ac.uk/group/corpora/large4/librispeech/12"
setname = "test-clean"
tokenizer = whisper.tokenizer.get_tokenizer(True, language="en")

features = {}
for speaker in os.listdir(os.path.join(base_path, setname)):
    spkpath = os.path.join(base_path, setname, speaker)
    for project in os.listdir(spkpath):
        fullpath = os.path.join(spkpath, project)
        with open(os.path.join(fullpath, "{}-{}.trans.txt".format(speaker, project))) as fin:
            for line in fin:
                uttname = line.split()[0]
                print(uttname)
                utt = " " + ' '.join(line.split()[1:])
                utttokens = tokenizer.encode(utt.lower())

                # Generate output path in writable directory, preserving structure
                output_fullpath = os.path.join(base_output_path, setname, speaker, project)
                os.makedirs(output_fullpath, exist_ok=True)  # Ensure the directory exists

                # Load audio and save mel spectrogram
                audiopath = os.path.join(base_path, fullpath, "{}.flac".format(uttname))
                audio = whisper.load_audio(audiopath)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio)

                # Save the .pt file to the writable directory
                dumppath = os.path.join(output_fullpath, "{}_fbank.pt".format(uttname))
                torch.save(mel, dumppath)
                datapiece = {"fbank": dumppath, "words": utt}
                features[uttname] = datapiece

# Save the metadata to JSON in /disk/data4 (in case I run out of AFS storage)
output_json_path = os.path.join(base_output_path, setname.replace("-", "_") + ".json")
with open(output_json_path, "w") as fout:
    json.dump(features, fout, indent=4)
