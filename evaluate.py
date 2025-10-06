# evaluation script for HMM chord recognition model
import data_utils
import random
import time
import numpy as np

ALL_ERAS = ["piano_addon", "piano_baroque", "piano_classical", "piano_modern", "piano_romantic", "orchestra_addon", "orchestra_baroque", "orchestra_classical", "orchestra_modern", "orchestra_romantic"]
CHORDS = ['A_aug', 'A_dim', 'A_dim_dim7', 'A_dim_min7', 'A_maj', 'A_maj_maj7', 'A_maj_min7', 'A_min', 'A_min_min7', 'Ab_dim', 'Ab_dim_min7', 'Ab_maj', 'Ab_maj_maj7', 'Ab_maj_min7', 'Ab_min', 'Ab_min_min7', 'B_aug', 'B_dim', 'B_dim_dim7', 'B_dim_min7', 'B_maj', 'B_maj_maj7', 'B_maj_min7', 'B_min', 'B_min_min7', 'Bb_aug', 'Bb_dim', 'Bb_dim_dim7', 'Bb_dim_min7', 'Bb_maj', 'Bb_maj_maj7', 'Bb_maj_min7', 'Bb_min', 'Bb_min_min7', 'C#_dim', 'C#_dim_min7', 'C#_maj', 'C#_maj_maj7', 'C#_maj_min7', 'C#_min', 'C#_min_min7', 'C_aug', 'C_dim', 'C_dim_min7', 'C_maj', 'C_maj_maj7', 'C_maj_min7', 'C_min', 'C_min_min7', 'D_dim', 'D_dim_min7', 'D_maj', 'D_maj_maj7', 'D_maj_min7', 'D_min', 'D_min_min7', 'E_dim', 'E_dim_min7', 'E_maj', 'E_maj_maj7', 'E_maj_min7', 'E_min', 'E_min_min7', 'Eb_dim', 'Eb_dim_min7', 'Eb_maj', 'Eb_maj_maj7', 'Eb_maj_min7', 'Eb_min', 'Eb_min_min7', 'F#_dim', 'F#_dim_min7', 'F#_maj', 'F#_maj_maj7', 'F#_maj_min7', 'F#_min', 'F#_min_min7', 'F_dim', 'F_dim_min7', 'F_maj', 'F_maj_maj7', 'F_maj_min7', 'F_min', 'F_min_min7', 'G_dim', 'G_dim_min7', 'G_maj', 'G_maj_maj7', 'G_maj_min7', 'G_min', 'G_min_min7', 'N']

def reformat_data(chord_data, frame_size = 0.1):
    formated_data = {}
    for era in ALL_ERAS:
        formated_data[era] = {}
    
    processed_files = 0
    for key in chord_data:
        val = chord_data[key]
        
        
        cur_time = val[0][0]
        cur_chord = val[1][0]
        n = len(val[0])
        next_time = val[0][1]
        next_chord = val[1][1]

        labels = []
        i = 1
        t = 0

        while next_time is not None:
            #Current chord is in majority in the current frame
            if next_time > t + frame_size/2:
                labels.append(cur_chord)
            else: #current chord may not be in majority
                best_chord = cur_chord
                best_time = next_time-t
                while next_time < t + frame_size:
                    cur_time = next_time
                    cur_chord = next_chord
                    i += 1
                    if i >= n: 
                        next_time = None
                        next_chord = None
                        break
                    next_time = val[0][i]
                    next_chord = val[1][i]

                    if min(t+frame_size, next_time) - cur_time > best_time:
                        best_chord = cur_chord
                        best_time = next_time - cur_time
                labels.append(best_chord)

                

            t += frame_size
        
        #TODO actually find the correct class here
        cls = random.choice(ALL_ERAS)
        formated_data[cls][key] = labels
        if processed_files % 200 == 0:
            print(processed_files)

        processed_files += 1
        if key == "CrossEra-0518_Shostakovich_symphony_no13_kariera_career.mp3_asdf":
            print(labels)
            deth = 1/0
    return formated_data

    
        



def load_model():
    return None
    #TODO use hmm_model.load
def run_model(model, data, frame_size = 0.1, genres = ALL_ERAS):
    #TODO change this 
    preds = {}
    for genre in genres:
        subset = data[genre]
        for key in subset:
            N = len(subset[key])
            
            res = [0] * N
            for i in range(len(res)):
                res[i] = random.choice(CHORDS)
            preds[key] = res

    return preds

def format_model_output(out):
    return out #TODO maybe need to add something here depending on how model is implemented
def eval(model, chord_data, classes = ALL_ERAS):
    predictions = run_model(model, chord_data) #TODO run on chromagram data when model is done

    conf_mat = np.zeros((len(CHORDS), len(CHORDS)))
    print("conf mat shape: ", conf_mat.shape)
    hits = 0
    misses = 0
    for cls in classes:
        for key in chord_data[cls]:
            annotation = chord_data[cls][key]
            pred = predictions[key]
            for i in range(len(annotation)):
                a_index = CHORDS.index(annotation[i])
                pred_index = CHORDS.index(pred[i])
                #print(pred_index, a_index)
                conf_mat[pred_index][a_index] += 1
                if a_index == pred_index:
                    hits += 1
                else:
                    misses += 1

    return hits, misses, conf_mat


def main():
    ## TODO: Implement evaluation pipeline
    # this script should load trained HMM models from disk,
    # evaluate them on test data, and generate comparison reports
    # including accuracy metrics, confusion matrices, and plots
    chord_data = data_utils.load_chord_annotations()

    t_before = time.time()
    chord_data = reformat_data(chord_data)

    print("data reformated in ", time.time()-t_before, "seconds")

    t_before = time.time()
    m = load_model()
    preds = run_model(m, chord_data)
    print("model ran in ", time.time()-t_before, "seconds")
    

    preds = format_model_output(preds)

    t_before = time.time()
    hits, misses, conf_mat = eval(m, chord_data)
    print("evaluation done in ", time.time()-t_before, "seconds")


    print("hits:" , hits)
    print("misses:", misses)
    print("conf_mat: ", conf_mat)
    #TODO load models
if __name__ == "__main__":
    main()