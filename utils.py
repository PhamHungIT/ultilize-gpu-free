import csv
from tqdm import tqdm


def write_csv(samples, labels, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['sample', 'label'])
        for sample, label in tqdm(zip(samples, labels), total=len(samples), desc="write file"):
            try:
                sample = sample.replace('\t', '')
                writer.writerow([sample, label])
            except:
                print(f'sample error: {sample}')


def write_csv_tuple(samples, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['sample', 'predicted', 'label'])
        for (sample, predicted_label, groundtruth) in tqdm(samples, total=len(samples), desc='write file'):
            writer.writerow([sample, predicted_label, groundtruth])


def write_csv_3col(raw_samples, samples, labels, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['raw_sample', 'sample', 'label'])
        for raw_sample, sample, label in tqdm(zip(raw_samples, samples, labels), total=len(samples), desc="write file"):
            try:
                sample = sample.replace('\t', '')
                writer.writerow([raw_sample, sample, label])
            except:
                print(f'sample error: {sample}')


def write_csv_intent_domain(samples, intents, domains, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['sample', 'label', 'domain'])
        for sample, intent, domain in tqdm(zip(samples, intents, domains), total=len(samples), desc="write file"):
            try:
                sample = sample.replace('\t', '')
                writer.writerow([sample, intent, domain])
            except:
                print(f'sample error: {sample}')


def write_csv_predicted(samples, labels, predicted_labels, probs, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['sample', 'label', 'predicted_label', 'probs'])
        for sample, label, predicted_label, prob in tqdm(zip(samples, labels, predicted_labels, probs), total=len(samples)):
            try:
                sample = sample.replace('\t', '')
                writer.writerow([sample, label, predicted_label, prob])
            except:
                print(f'sample error: {sample}')


def write_csv_mask_predicted(samples, mask_samples, labels, predicted_labels, probs, csv_file):
    with open(csv_file, 'w') as fo:
        writer = csv.writer(fo, delimiter='\t')
        writer.writerow(['sample', 'mask_sample', 'label', 'predicted_label', 'probs'])
        for sample, mask_sample, label, predicted_label, prob in tqdm(zip(samples, mask_samples, labels, predicted_labels, probs), total=len(samples)):
            try:
                sample = sample.replace('\t', '')
                writer.writerow([sample, mask_sample, label, predicted_label, prob])
            except:
                print(f'sample error: {sample}')


def write_txt(samples, outfile):
    with open(outfile, 'w') as fw:
        for sample in samples:
            fw.write(sample.strip() + "\n")