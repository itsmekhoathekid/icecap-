from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
# python3
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def language_eval(data_set, preds, model_id, split):
    import sys
    if data_set == 'breakingnews':
        if split == 'val':
            annFile = './breakingnews_data/breakingnews_val.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)
        else:
            annFile = './breakingnews_data/breakingnews_test.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)

    elif data_set == 'goodnews':
        if split == 'val':
            annFile = './goodnews_data/goodnews_val.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)
        else:
            annFile = './goodnews_data/goodnews_test.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)
    elif data_set == 'ViWiki':
        if split == 'val':
            annFile = '/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/ViWiki_val.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)
        else:
            annFile = '/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/ViWiki_test.json'
            print('annFile', annFile)
            with open(annFile, 'rb') as f:
                dataset = json.load(f)


    id_to_ix = {v['cocoid'].split('_')[0]: ix for ix, v in enumerate(dataset)}  # cocoid > index
    hypo = {v['image_id'].split('_')[0]: [v['caption']] for v in preds}  # cocoid > predicted_caption
    ref = {k: [i['raw'] for i in dataset[id_to_ix[k]]['sentences']] for k in hypo.keys()}  # cocoid > true_caption
    final_scores = evaluate(ref, hypo)
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])
    print('Bleu_3:\t', final_scores['Bleu_3'])
    print('Bleu_4:\t', final_scores['Bleu_4'])
    print('METEOR:\t', final_scores['METEOR'])
    print('ROUGE_L:', final_scores['ROUGE_L'])
    print('CIDEr:\t', final_scores['CIDEr'])
    return final_scores


def att_true_count(att_output, att_labels, att_mask):
    """

    :param att_output: batch * seq_length * max_word_length
    :param att_labels: batch * seq_length * gold_num
    :param att_mask: batch * seq_length * gold_num
    :return:
    """
    att_labels = att_labels[:, :att_output.size(1)]
    att_mask = att_mask[:, :att_output.size(1)]
    target_num = torch.nonzero(torch.sum(att_mask, dim=2)).size()[0]
    output_index = torch.argmax(att_output, dim=2)
    # print(output_index.size())
    # print(att_labels.size())
    output_index = output_index.unsqueeze(2).expand_as(att_labels) + 1
    diff = output_index - att_labels
    zero_num = att_labels.numel() - torch.nonzero(diff).size()[0]
    true_num = zero_num
    return target_num, true_num


def evaluate(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def eval_split(cnn_model, model, crit, loader, eval_kwargs={}, return_attention=False, return_w_attention=False):
    verbose = eval_kwargs.get('verbose', False)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'news')
    sen_init = eval_kwargs.get('sen_init', False)
    img_init = eval_kwargs.get('img_init', False)
    sim_sen_init = eval_kwargs.get('sen_sim_init', False)
    word_embed_att = eval_kwargs.get('word_embed_att', False)
    word_mask = eval_kwargs.get('word_mask', False)
    caption_model = eval_kwargs.get('caption_model', '')
    index_size = eval_kwargs.get('index_size', -1)
    pointer_matching = eval_kwargs.get('pointer_matching', False)

    print('evaluating...')
    print('sen init', sen_init)
    print('img init', img_init)
    print('sim sen init', sim_sen_init)
    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    if pointer_matching:
        cap_loss_sum = 0
        match_loss_sum = 0
        match_all_target_num = 0.0
        match_all_true_num = 0.0
    while True:
        data = loader.get_batch(split)
        data['images'] = utils.prepro_images(data['images'], False)
        n = n + loader.batch_size

        # evaluate loss if we have the labels
        loss = 0
        # Get the image features first
        tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp
        with torch.no_grad():
            att_feats = cnn_model(images).permute(0, 2, 3, 1)  # .contiguous()
            fc_feats = att_feats.mean(2).mean(1)
        sen_embed = data.get('sen_embed', None)
        sim_sen = data.get('sim', None)
        sim_words = data.get('sim_words', None)
        word_masks = data.get('word_masks', None)
        match_labels = data.get('match_labels', None)
        match_masks = data.get('match_masks', None)
        sim_words_index = data.get('sim_words_index', None)

        # forward the model to get loss
        if data.get('labels', None) is not None:
            att_feats = att_feats.unsqueeze(1).expand(
                *((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(
                *((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(
                *((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(
                *((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))
            if sen_embed is not None:
                sen_embed = Variable(torch.from_numpy(np.array(sen_embed)).cuda())
                if sim_sen_init:
                    sim_sen = Variable(torch.from_numpy(np.array(sim_sen))).cuda()
                else:
                    sim_sen = None
                if word_embed_att:
                    similar_words = Variable(torch.from_numpy(np.array(sim_words))).cuda()
                    if word_mask:
                        word_masks = Variable(torch.from_numpy(np.array(word_masks))).cuda()
                    else:
                        word_masks=None
                    if pointer_matching:
                        match_labels = Variable(torch.from_numpy(np.array(match_labels))).cuda()
                        match_masks = Variable(torch.from_numpy(np.array(match_masks))).cuda()
                        # similar_words_index = Variable(torch.from_numpy(np.array(sim_words_index))).cuda()
                    if index_size != -1:
                        similar_words_index = Variable(torch.from_numpy(np.array(sim_words_index))).cuda()
                    else:
                        similar_words_index = None
                else:
                    similar_words_index = None
                    similar_words = None
                    word_masks = None

                with torch.no_grad():
                    if pointer_matching:
                        output, match_output = model(fc_feats, att_feats, labels, sen_embed, similar_words,
                                                     word_masks, sim_sen, sen_init, img_init,
                                                     None, similar_words_index)
                        cap_loss, match_loss = crit(output, labels[:, 1:], masks[:, 1:], match_output, match_labels, match_masks)
                        match_target_num, match_true_num = att_true_count(match_output, match_labels, match_masks)
                        loss = cap_loss + match_loss
                    else:
                        loss = crit(model(fc_feats, att_feats, labels, sen_embed, similar_words, word_masks, sim_sen, sen_init, img_init, None, similar_words_index),
                                    labels[:, 1:], masks[:, 1:])
            else:
                if 'show_attend_tell' in caption_model:
                    loss = crit(model(fc_feats, att_feats, labels, img_init=img_init), labels[:, 1:], masks[:, 1:]).data[0]
                else:
                    loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:]).data[0]
            if pointer_matching:
                cap_loss_sum += cap_loss
                match_loss_sum += match_loss
                match_all_target_num += match_target_num
                match_all_true_num += match_true_num
            loss_sum += loss
            loss_evals = loss_evals + 1

        if sen_embed is not None:
            if return_attention or return_w_attention:
                seq, _, atts = model.sample(fc_feats, att_feats, eval_kwargs, sen_embed, similar_words, word_masks, sim_sen, similar_words_index, return_attention, return_w_attention)
                vis_attention = np.array([att[0] for att in atts])
                sen_attention = np.array([att[1] for att in atts]) #
            # else:
            if pointer_matching:
                seq, _, match_seqprobs = model.sample(fc_feats, att_feats, eval_kwargs, sen_embed,
                                                            similar_words, word_masks,
                                                            sim_sen, similar_words_index, return_attention,
                                                            return_w_attention)
                #################################
                # per_example = []
                # for seq_list in match_seqprobs:
                #     steps = [np.array(step) for step in seq_list]
                #     Ps = [step.shape[0] for step in steps]
                #     Vs = [step.shape[1] for step in steps]
                #     max_P, max_V = max(Ps), max(Vs)
                #     padded_steps = []
                #     for step in steps:
                #         P, V = step.shape
                #         pad_rows = max_P - P
                #         pad_cols = max_V - V
                #         padded = np.pad(
                #             step,
                #             ((0, pad_rows), (0, pad_cols)),
                #             mode='constant',
                #             constant_values=0
                #         )
                #         padded_steps.append(padded)
                #     per_example.append(np.stack(padded_steps, axis=0))
                # batch_size = len(per_example)
                # P, V = per_example[0].shape[1], per_example[0].shape[2]
                # max_T = max(arr.shape[0] for arr in per_example)

                # padded = np.zeros((batch_size, max_T, P, V), dtype=per_example[0].dtype)
                # for i, arr in enumerate(per_example):
                #     T_i = arr.shape[0]
                #     padded[i, :T_i, :, :] = arr
                # match_seqprobs = padded #(batch_size, max_T, P, V) batch, the padded sentence length, Position, vocab

                per_time = []
                for step in match_seqprobs:
                    flat_parts = []
                    for pos_vec in step:
                        arr = np.array(pos_vec).reshape(-1)
                        flat_parts.append(arr)
                    per_time.append(np.concatenate(flat_parts, axis=0))


                #################################
                # match_seqprobs = np.array(match_seqprobs)
            else:
                seq, _ = model.sample(fc_feats, att_feats, eval_kwargs, sen_embed, similar_words, word_masks, sim_sen, similar_words_index, return_attention, return_w_attention)
        else:
            seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        # set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        #
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'image_path': data['infos'][k]['file_path']}
            if pointer_matching:
                sen_length = len(sent.split())
                # match_seqprobs_sent = np.array(match_seqprobs[:sen_length, k, :].tolist())  # sent_len * word_length
                ###################
                # per_time = [np.array(step) for step in match_seqprobs]   
                match_seqprobs_sent = np.stack(per_time[:sen_length], axis=0)
                ###################
                sorted_index = [np.argsort(match_logprobs)[::-1].tolist() for match_logprobs in match_seqprobs_sent]
                top_match_weight = [np.sort(match_logprobs)[-10:][::-1].tolist() for match_logprobs in match_seqprobs_sent]
                entry['match_index'] = sorted_index
                entry['top_match_weight'] = top_match_weight

            if return_attention:
                sen_length = len(sent.split())
                entry['vis_att'] = vis_attention[:sen_length, k, :].tolist()
                # sen_attention: max_seq_len * batch * max_sen_num
                entry['sen_att'] = sen_attention[:sen_length, k, :].tolist()
            elif return_w_attention:
                # print(sen_attention.shape)
                sen_length = len(sent.split())
                # sen_attention: max_seq_len * batch * max_word_num
                attention = sen_attention[:sen_length, k, :].tolist()  # sent_len*max_word_num
                # print(np.array(attention).shape)
                attention = np.array(attention)
                # print(attention.shape)
                # exit(0)
                top_att_weight = [np.sort(a)[-10:][::-1].tolist() for a in attention]
                sorted_att_loc = [np.argsort(a)[-200:][::-1].tolist() for a in attention]
                entry['sorted_word_att'] = sorted_att_loc  # sent_len * max_word_num
                entry['top_att_weight'] = top_att_weight  # sent_len * 10

            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        # break
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    if pointer_matching:
        print('cap loss:%.3f, match_loss:%.3f' % (cap_loss_sum / loss_evals, match_loss_sum / loss_evals))
        print('match precision:%.3f' % (float(match_all_true_num) / match_all_target_num))

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats

# def eval_split(cnn_model, model, crit, loader, eval_kwargs={}, return_attention=False, return_w_attention=False):
#     import numpy as np
#     verbose = eval_kwargs.get('verbose', False)
#     num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
#     split = eval_kwargs.get('split', 'val')
#     lang_eval = eval_kwargs.get('language_eval', 0)
#     dataset = eval_kwargs.get('dataset', 'news')
#     sen_init = eval_kwargs.get('sen_init', False)
#     img_init = eval_kwargs.get('img_init', False)
#     sim_sen_init = eval_kwargs.get('sen_sim_init', False)
#     word_embed_att = eval_kwargs.get('word_embed_att', False)
#     word_mask = eval_kwargs.get('word_mask', False)
#     caption_model = eval_kwargs.get('caption_model', '')
#     index_size = eval_kwargs.get('index_size', -1)
#     pointer_matching = eval_kwargs.get('pointer_matching', False)

#     print('evaluating...')
#     print('sen init', sen_init)
#     print('img init', img_init)
#     print('sim sen init', sim_sen_init)

#     cnn_model.eval()
#     model.eval()
#     loader.reset_iterator(split)

#     n = 0
#     loss_sum = 0
#     loss_evals = 1e-8
#     predictions = []
#     cap_loss_sum = match_loss_sum = 0.0
#     match_all_target_num = match_all_true_num = 0.0

#     while True:
#         data = loader.get_batch(split)
#         data['images'] = utils.prepro_images(data['images'], False)
#         n += loader.batch_size

#         # Prepare inputs
#         tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
#         tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
#         images, labels, masks = tmp
#         with torch.no_grad():
#             att_feats = cnn_model(images).permute(0, 2, 3, 1)
#             fc_feats = att_feats.mean(2).mean(1)

#         # Unpack optional inputs for pointer matching
#         sen_embed = data.get('sen_embed', None)
#         sim_sen = data.get('sim', None)
#         sim_words = data.get('sim_words', None)
#         word_masks = data.get('word_masks', None)
#         match_labels = data.get('match_labels', None)
#         match_masks = data.get('match_masks', None)
#         sim_words_index = data.get('sim_words_index', None)

#         # Compute loss if labels provided
#         if data.get('labels', None) is not None:
#             # Expand features for captions
#             att_feats = att_feats.unsqueeze(1).expand(
#                 *((att_feats.size(0), loader.seq_per_img) + att_feats.size()[1:])
#             ).contiguous().view(
#                 *((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:])
#             )
#             fc_feats = fc_feats.unsqueeze(1).expand(
#                 *((fc_feats.size(0), loader.seq_per_img) + fc_feats.size()[1:])
#             ).contiguous().view(
#                 *((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:])
#             )

#             # Convert additional inputs to tensors
#             if sen_embed is not None:
#                 sen_embed = Variable(torch.from_numpy(np.array(sen_embed))).cuda()
#                 if sim_sen_init:
#                     sim_sen = Variable(torch.from_numpy(np.array(sim_sen))).cuda()
#                 else:
#                     sim_sen = None
#                 if word_embed_att:
#                     similar_words = Variable(torch.from_numpy(np.array(sim_words))).cuda()
#                     word_masks = Variable(torch.from_numpy(np.array(word_masks))).cuda() if word_mask else None
#                     if pointer_matching:
#                         match_labels = Variable(torch.from_numpy(np.array(match_labels))).cuda()
#                         match_masks = Variable(torch.from_numpy(np.array(match_masks))).cuda()
#                     similar_words_index = (
#                         Variable(torch.from_numpy(np.array(sim_words_index))).cuda()
#                         if index_size != -1 else None
#                     )
#                 else:
#                     similar_words = similar_words_index = word_masks = None

#                 with torch.no_grad():
#                     if pointer_matching:
#                         output, match_output = model(
#                             fc_feats, att_feats, labels, sen_embed,
#                             similar_words, word_masks, sim_sen,
#                             sen_init, img_init, None, similar_words_index
#                         )
#                         cap_loss, match_loss = crit(
#                             output, labels[:,1:], masks[:,1:],
#                             match_output, match_labels, match_masks
#                         )
#                         loss = cap_loss + match_loss
#                         match_target_num, match_true_num = att_true_count(
#                             match_output, match_labels, match_masks
#                         )
#                         cap_loss_sum += cap_loss
#                         match_loss_sum += match_loss
#                         match_all_target_num += match_target_num
#                         match_all_true_num += match_true_num
#                     else:
#                         loss = crit(
#                             model(
#                                 fc_feats, att_feats, labels,
#                                 sen_embed, similar_words, word_masks,
#                                 sim_sen, sen_init, img_init,
#                                 None, similar_words_index
#                             ),
#                             labels[:,1:], masks[:,1:]
#                         )
#             else:
#                 loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])

#             loss_sum += loss
#             loss_evals += 1

#         # Sampling and pointer-matching postprocessing
#         if sen_embed is not None and pointer_matching:
#             seq, _, match_seqprobs = model.sample(
#                 fc_feats, att_feats, eval_kwargs, sen_embed,
#                 similar_words, word_masks,
#                 sim_sen, similar_words_index,
#                 return_attention, return_w_attention
#             )
#             # → CHỈ ĐOẠN SỬA Ở ĐÂY ←
#             # Stack theo time-step để có array (time, batch, word_len), rồi transpose
#             for idx, arr in enumerate(match_seqprobs):
#                 print(f"Shape of array {idx}: {np.array(arr).shape}")
#             arr = np.stack(match_seqprobs, axis=0)
#             match_seqprobs = arr.transpose(1, 0, 2)
#             print("calculated match_seqprobs")

#         else:
#             if sen_embed is not None:
#                 seq, _ = model.sample(
#                     fc_feats, att_feats, eval_kwargs,
#                     sen_embed, similar_words, word_masks,
#                     sim_sen, similar_words_index,
#                     return_attention, return_w_attention
#                 )
#             else:
#                 seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

#         sents = utils.decode_sequence(loader.get_vocab(), seq)

#         # Build predictions entries
#         for k, sent in enumerate(sents):
#             entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
#             if pointer_matching:
#                 L = len(sent.split())
#                 slice_k = match_seqprobs[k, :L, :]
#                 sorted_index = [np.argsort(logprobs)[::-1].tolist() for logprobs in slice_k]
#                 top_match_weight = [np.sort(logprobs)[-10:][::-1].tolist() for logprobs in slice_k]
#                 entry['match_index'] = sorted_index
#                 entry['top_match_weight'] = top_match_weight
#             predictions.append(entry)

#         # Check iteration bounds
#         ix0 = data['bounds']['it_pos_now']
#         ix1 = min(data['bounds']['it_max'], num_images if num_images != -1 else data['bounds']['it_max'])
#         if data['bounds']['wrapped'] or (num_images >= 0 and n >= num_images):
#             break

#     lang_stats = None
#     if lang_eval == 1:
#         lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
#     if pointer_matching:
#         print(f"cap loss:{cap_loss_sum/loss_evals:.3f}, match_loss:{match_loss_sum/loss_evals:.3f}")
#         print(f"match precision:{(match_all_true_num/match_all_target_num if match_all_target_num else 0):.3f}")

#     model.train()
#     return loss_sum / loss_evals, predictions, lang_stats
