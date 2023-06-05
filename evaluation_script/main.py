import random
import json 
import numpy as np

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}

    # test_videos = json.load(open('/private/home/afourast/ht100m-step/annotations_v0/test_videos.json'))
    # annots = json.load(open('/private/home/afourast/ht100m-step/annotations_v0/annotations_formatted.json'))
    # test_annots = {ann['video']: ann for ann in annots if ann['video'] in test_videos}
    # with open('/private/home/afourast/ht100m-step/annotations_v0/test_annotations_formatted.json', 'w') as fw:
    #     json.dump(test_annots, fw)

    # test_annots = json.load(open('/private/home/afourast/ht100m-step/annotations_v0/test_annotations_formatted.json'))
    # submission = json.load(open('/private/home/afourast/ht100mstep-challenge/test_outputs/recall_1.json'))
    # test_annots_5 = {kk: test_annots[kk] for kk in sorted(test_annots.keys())[:5]}
    # with open('/private/home/afourast/ht100m-step/annotations_v0/test_annotations_formatted_5.json', 'w') as fw:
    #     json.dump(test_annots_5, fw)

    test_annots = json.load(open(test_annotation_file))
    submission = json.load(open(user_submission_file))

    if not len(submission) == len(test_annots):
        print('Missing some annotations')

    recalls = []

    for vid in submission:

        if vid not in test_annots:
            continue

        n_txt = len(submission[vid])
        gt = test_annots[vid]
        assert len(gt['segments']) == n_txt

        for text_idx in range(n_txt):

            if not gt['aligned'][text_idx]:
                continue

            pred_timestamp = submission[vid][text_idx]

            segments_aligned = gt['segments'][text_idx]
            retrieved = False
            for segment in segments_aligned:
                if segment[0] <= pred_timestamp <= segment[1]:
                    retrieved = True
                    break
            recalls.append(retrieved)

    recall = np.mean(recalls)

    output["result"] = [
        {
            "test_split": {
                "recall@1": recall,
            }
        }
    ]

    print(output)

    return output
