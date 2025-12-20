# -*- coding: utf-8 -*-
import ke
import hydra
import torch
import numpy as np
import sklearn
from tqdm import tqdm
from torchinfo import summary

logger = ke.logger

@hydra.main(version_base=None, config_path='ke/conf', config_name='config')
def main(config):
    logger.info(f'Using Config')
    ke.utils.resolve_to_environ(config)
    logger.info(f'{config.data}')
    logger.info(f'{config.model}')
    logger.info(f'{config.hyper}')
    # [logger.info(i) for i in json.dumps(json.loads(config.__str__().replace('\'','"').replace('True','true').replace('False','false')),indent=4).__str__().split('\n')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {torch.cuda.get_device_name(0)}')
    logger.info(f'Load Data {config.data.name} ...')
    
    ner_processor = ke.utils.NERProcessor(config)

    train_loader = torch.utils.data.DataLoader(
        dataset     = ner_processor.train_examples,
        sampler     = torch.utils.data.RandomSampler(ner_processor.train_examples),
        batch_size  = config.hyper.batch_size,
        num_workers = config.hyper.num_workers,
        collate_fn  = ner_processor.collate,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset     = ner_processor.valid_examples,
        sampler     = torch.utils.data.SequentialSampler(ner_processor.valid_examples),
        batch_size  = config.hyper.batch_size,
        num_workers = config.hyper.num_workers,
        collate_fn  = ner_processor.collate,
    )

    model = {
        'BiLSTM':                   ke.models.BiLSTM_CRF,
        'BicLSTM':                  ke.models.BicLSTM_CRF,
        'BiLSTM_Small':             ke.models.BiLSTM_CRF,
        'BicLSTM_Small':            ke.models.BicLSTM_CRF,
        'BiLSTM_Large':             ke.models.BiLSTM_CRF,
        'BicLSTM_Large':            ke.models.BicLSTM_CRF,
    }[config.model.name](ner_processor, config.model).to(device)
    logger.info(f'Load Model {config.model.name}.')

    optimizer = torch.optim.Adam(
        params  = model.parameters(), 
        lr      = config.hyper.learning_rate, 
        betas   = (config.hyper.beta1, config.hyper.beta2)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer = optimizer,
        step_size = config.hyper.lr_step,
        gamma     = config.hyper.lr_gamma
    )

    one_batch = (x.to(device) for x in next(iter(train_loader)))
    i, _, m = one_batch

    logger.info('')

    [
        logger.info(i) 
        for i in summary(
            model=model,
            input_data = (i,m),
            verbose=False
        ).__str__().split('\n')
    ]

    logger.info('')
    
    logger.info(f'Start Task {config.task.capitalize()} ...')
    if config.task == 'train':
        model_src = config.model.save_path
        if config.train:
            max_avg_f1= 0
            for epoch in range(config.hyper.num_epochs):
                model.train()
                tr_loss = 0
                num_training_steps = len(train_loader)
                progress_bar = tqdm(range(num_training_steps), total=num_training_steps, leave=False)
                for step, batch in enumerate(train_loader):
                    input, target, mask = (x.to(device) for x in batch)
                    y_pred              = model(input, mask)
                    loss                = model.loss_fn(input, target, mask)
                    loss.backward()

                    tr_loss            += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                progress_bar.close()
                scheduler.step()
                logger.info(f'>> Train: [{epoch+1:4d}/{config.hyper.num_epochs}]:\t Loss: {(tr_loss / len(train_loader)):.4f}')
                

                with torch.no_grad():
                    model.eval()
                    y_true_list = []
                    y_pred_list = []
                    valid_loss = 0
                    num_valid_steps = len(valid_loader)
                    progress_bar = tqdm(enumerate(valid_loader), total=num_valid_steps, leave=False)
                    for step, batch in enumerate(valid_loader):
                        input, target, mask = (x.to(device) for x in batch)
                        y_pred              = model(input, mask)
                        loss                = model.loss_fn(input, target, mask)
                        for lst in y_pred:
                            y_pred_list += lst
                        for y,m in zip(target, mask):
                            y_true_list += y[m == True].tolist()
                        valid_loss += loss.item()
                        progress_bar.update(1)
                    progress_bar.close()
                    
                    eval_labels = list(ner_processor.l2i.keys()).copy()
                    eval_labels.remove('O')

                    assert (np.array(y_true_list)).shape == (np.array(y_pred_list)).shape
                    assert (np.array(y_true_list)).ndim == 1
                    assert (np.array(y_pred_list)).ndim == 1

                    valid_f1 = sklearn.metrics.f1_score(
                        [ ner_processor.i2l[l] for l in y_true_list ], 
                        [ ner_processor.i2l[l] for l in y_pred_list ],
                        labels=eval_labels, average='weighted'
                    )

                    valid_loss /= len(valid_loader)

                    logger.info(f'>> Valid: [{epoch+1:4d}/{config.hyper.num_epochs}]:\t Loss: {valid_loss:.4f}\t F1: {valid_f1:.4f}')

                    if max_avg_f1 < valid_f1:
                        max_avg_f1 = valid_f1
                        logger.info(f'>> Save Model with\t\t Loss: {valid_loss:.4f}\t F1: {valid_f1:.4f}')
                        model_src = model.save(epoch+1)

        if config.valid:
            valid_model = model.load(path=model_src, device=device)
            valid_model = valid_model.to(device)
            valid_model.eval()

            y_true_list, y_pred_list = [], []
            valid_loss = 0
            num_valid_steps = len(valid_loader)
            progress_bar = tqdm(enumerate(valid_loader), total=num_valid_steps, leave=False)
            for step, batch in enumerate(valid_loader):
                input, target, mask = (x.to(device) for x in batch)
                y_pred              = model(input, mask)
                loss                = model.loss_fn(input, target, mask)
                for lst in y_pred:
                    y_pred_list += lst
                for y,m in zip(target, mask):
                    y_true_list += y[m == True].tolist()
                valid_loss += loss.item()
                progress_bar.update(1)
            progress_bar.close()

            report = sklearn.metrics.classification_report(
                [ner_processor.i2l[i] for i in y_true_list],
                [ner_processor.i2l[i] for i in y_pred_list],
                digits=4
            )

            logger.info(f'>> Total: {len(valid_loader)}\t Loss: {valid_loss:.4f}')
            [ logger.info(i) for i in report.split('\n') ]

    elif config.task == 'predict':
        valid_model = model.load(path=config.model.save_path, device=device)
        valid_model = valid_model.to(device)
        valid_model.eval()

        words   = torch.tensor([[ ner_processor.w2i.get(i, 0) for i in config.text ]], device = device)
        mask    = torch.tensor([ [True] * len(words[0]) ], device = device)
        y_pred  = valid_model(words, mask)

        words   = [ w for w in config.text ]
        labels  = [ ner_processor.i2l[i] for i in y_pred[0]]

        result, results = [], []
        for word, label in zip(words, labels):
            if label != 'O':
                result.append((word,label))
            else:
                if len(result) > 0:
                    results.append(result)
                    result = []

        results = [ (''.join([j[0] for j in i]), i[0][1].split('-')[1]) for i in results ]

        logger.info(f'>> Origin: {config.text}')
        logger.info(f'>> Result: {results}')

    else:
        logger.error(f'Task {config.task} is not supported.')

if __name__ == '__main__':
    main()