# evaluation script for HMM chord recognition model
import data_utils
import random
import time
import numpy as np
import matplotlib.pyplot as plt

ALL_ERAS = ["piano_addon", "piano_baroque", "piano_classical", "piano_modern", "piano_romantic", "orchestra_addon", "orchestra_baroque", "orchestra_classical", "orchestra_modern", "orchestra_romantic"]
CHORDS = ['A_aug', 'A_dim', 'A_dim_dim7', 'A_dim_min7', 'A_maj', 'A_maj_maj7', 'A_maj_min7', 'A_min', 'A_min_min7', 'Ab_dim', 'Ab_dim_min7', 'Ab_maj', 'Ab_maj_maj7', 'Ab_maj_min7', 'Ab_min', 'Ab_min_min7', 'B_aug', 'B_dim', 'B_dim_dim7', 'B_dim_min7', 'B_maj', 'B_maj_maj7', 'B_maj_min7', 'B_min', 'B_min_min7', 'Bb_aug', 'Bb_dim', 'Bb_dim_dim7', 'Bb_dim_min7', 'Bb_maj', 'Bb_maj_maj7', 'Bb_maj_min7', 'Bb_min', 'Bb_min_min7', 'C#_dim', 'C#_dim_min7', 'C#_maj', 'C#_maj_maj7', 'C#_maj_min7', 'C#_min', 'C#_min_min7', 'C_aug', 'C_dim', 'C_dim_min7', 'C_maj', 'C_maj_maj7', 'C_maj_min7', 'C_min', 'C_min_min7', 'D_dim', 'D_dim_min7', 'D_maj', 'D_maj_maj7', 'D_maj_min7', 'D_min', 'D_min_min7', 'E_dim', 'E_dim_min7', 'E_maj', 'E_maj_maj7', 'E_maj_min7', 'E_min', 'E_min_min7', 'Eb_dim', 'Eb_dim_min7', 'Eb_maj', 'Eb_maj_maj7', 'Eb_maj_min7', 'Eb_min', 'Eb_min_min7', 'F#_dim', 'F#_dim_min7', 'F#_maj', 'F#_maj_maj7', 'F#_maj_min7', 'F#_min', 'F#_min_min7', 'F_dim', 'F_dim_min7', 'F_maj', 'F_maj_maj7', 'F_maj_min7', 'F_min', 'F_min_min7', 'G_dim', 'G_dim_min7', 'G_maj', 'G_maj_maj7', 'G_maj_min7', 'G_min', 'G_min_min7', 'N']

N24_RULEMAP = {
    "_dim7": "",
    "_maj7": "",
    "_min7": "",
    "_dim": None,
    "_aug": None
}

def gen_chords_from_rules(rule_map, chord_list = CHORDS):
    chords = chord_list[:]
    for rule in rule_map.keys():
        sub = rule_map[rule]
        for i in range(len(chords)):
            chord = chords[i]
            if chord is not None:
                if rule in chord:
                    if sub is None:
                        chords[i] = None
                    else:
                        chords[i] = chord.replace(rule, sub)
    chord_map = {
        CHORDS[i]: chords[i] for i in range(len(chords))
    }
    chords = list(set(chords)) #Removing duplicates
    if None in chords:
        chords.remove(None)
        chords.sort()   
        chords.append(None)
    else:
        chords.sort()
    

    return chords, chord_map


def reformat_data(chord_data, chord_map = None, frame_size = 0.1):
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
                if chord_map is None:
                    labels.append(cur_chord)
                else:
                    labels.append(chord_map[cur_chord])
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
                if chord_map is None:
                    labels.append(best_chord)
                else:
                    labels.append(chord_map[best_chord])

                

            t += frame_size
        
        #TODO actually find the correct class here
        cls = random.choice(ALL_ERAS)
        formated_data[cls][key] = labels

        processed_files += 1
        if key == "CrossEra-0518_Shostakovich_symphony_no13_kariera_career.mp3asdf":
            print(labels)
            deth = 1/0
    return formated_data

    



class Model:
    def __init__(self, name):
        self.name = name

def load_model(name):
    return Model(name)
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
def eval(model, chord_data, chord_list = CHORDS, chord_map = None, classes = ALL_ERAS):
    predictions = run_model(model, chord_data) #TODO run on chromagram data when model is done

    n = len(chord_list)
    conf_mat = np.zeros((n,n))
    #   Conf_mat[i,:] = row i, all values predicted as label i
    #   Confmat[:,i] = column i, all values that actually are label i
    hits = 0
    misses = 0
    for cls in classes:
        for key in chord_data[cls]:
            annotation = chord_data[cls][key]
            pred = predictions[key]
            if chord_map is not None:
                pred = [chord_map[pred[i]] for i in range(len(pred))]
            for i in range(len(annotation)):
                
                a_index = chord_list.index(annotation[i])
                pred_index = chord_list.index(pred[i])
                conf_mat[pred_index][a_index] += 1
                if a_index == pred_index:
                    hits += 1
                else:
                    misses += 1


    tot = hits+misses
    TP = np.zeros(n)
    FP = np.zeros(n)
    TN = np.zeros(n)
    FN = np.zeros(n)
    P = np.zeros(n)
    R = np.zeros(n)
    F1 = np.zeros(n)
    for i in range(n):
        TP[i] = conf_mat[i,i]
        FP[i] = np.sum(conf_mat[i, :]) - TP[i]
        FN[i] = np.sum(conf_mat[:, i]) - TP[i]
        TN[i] = tot - TP[i] - FP[i] - FN[i]
        P[i] = TP[i]/(TP[i] + FP[i])
        R[i] = TP[i] /(TP[i] +FN[i])
        F1[i] = 2*TP[i]/(2*TP[i] + FP[i] + FN[i])


    res = {}
    res["name"] = model.name
    res["hits"] = hits
    res["misses"] = misses
    res["conf_mat"] = conf_mat
    res["TP"] = TP
    res["FP"] = FP
    res["TN"] = TN
    res["FN"] = FN
    res["P"] = P
    res["R"] = R
    res["F1"] = F1



    return res


def compare_plot(reports, target_metrics, name = None, save = True):
    if name == None:
        name = "compare_plot"
    for metric in target_metrics:
        plt.clf()
        bar_vals = []
        bar_names = []
        for report in reports:
            bar_names.append(report["name"])
            if isinstance(report[metric], (int, float)):
                bar_vals.append(report[metric])
            else:
                bar_vals.append(np.mean(report[metric]))

        plt.bar(bar_names, bar_vals)
        plt.ylabel(metric)
        if save:
            plt.savefig("results/" + name + "_" + metric)
        else:
            plt.show()

def chord_compare(report, target_metrics, chord_list = CHORDS, target_chords = CHORDS, name = None, save = True):
    if name == None:
        name = "chord_compare_plot"
    for metric in target_metrics:
        plt.clf()
        if isinstance(report[metric], (int, float)):
            raise Exception("Cannot generate a per-chord comparison for a metric that is not a per-chord list")
        bar_vals = []
        bar_names = []
        for i in range(len(chord_list)):
            chord = chord_list[i]
            if chord in target_chords:
                bar_names.append(chord)
                bar_vals.append(report[metric][i])

        plt.bar(bar_names, bar_vals)
        plt.ylabel(metric)
        if save:
            plt.savefig("results/" + name + "_" + metric)
        else:
            plt.show()
        
    
def get_report(model, chord_data, chord_list = CHORDS, chord_map = None):
    t_before = time.time()

    preds = run_model(model, chord_data)
    print("model",model.name,"ran in ", time.time()-t_before, "seconds")
    

    preds = format_model_output(preds)

    t_before = time.time()
    report = eval(model, chord_data, chord_list=chord_list, chord_map=chord_map)
    print("evaluation done in ", time.time()-t_before, "seconds")
    return report


def main():
    ## TODO: Implement evaluation pipeline
    # this script should load trained HMM models from disk,
    # evaluate them on test data, and generate comparison reports
    # including accuracy metrics, confusion matrices, and plots

    #TODO maybe figure out how to use cache here.
    chord_data = data_utils.load_chord_annotations()

    #simple_chords, simple_map = gen_chords_from_rules(N24_RULEMAP)
    simple_chords, simple_map = gen_chords_from_rules({})
    
    #chrom_data = data_utils.load_chroma_features()
    t_before = time.time()
    chord_data = reformat_data(chord_data, simple_map)

    print("data reformated in ", time.time()-t_before, "seconds")

    t_before = time.time()
    m1 = load_model("m1")
    m2 = load_model("m2")
    m3 = load_model("m3")
    m4 = load_model("m4")
    models = [m1, m2, m3, m4]
    
    reports = []

    for m in models:

        #report = get_report(m, chord_data, simple_chords, simple_map)
        report = get_report(m, chord_data)
        reports.append(report)

    compare_plot(reports, ["TP", "F1"])
    chord_compare(reports[0], ["TP", "F1"], ["A_maj", "B_maj", "C_maj"])


if __name__ == "__main__":
    main()