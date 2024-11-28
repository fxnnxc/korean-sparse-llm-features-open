def get_tokenized_dataset(dataset, tokenizer, batch_size, num_proc=None, 
                          target_name='text', 
                          output_name='input_ids',
                          preserve_names=[],
                          max_length=None,
                          padding=True,
                          truncation=False):
    def process(samples):
        batch_inputs = {}    
        tokenized = tokenizer(samples[target_name], 
                                max_length=max_length, 
                                padding=padding, 
                                truncation=truncation, 
                                return_tensors='pt')     
        
        batch_inputs[output_name] = tokenized['input_ids']
        batch_inputs['attention_mask'] = tokenized['attention_mask']
        
        for v in preserve_names:
            batch_inputs[v] = samples[v]
        
        if 'label' in samples and isinstance(samples['label'][0], str):
            batch_inputs['label_ids'] = tokenizer(samples['label'], 
                                            max_length=max_length, 
                                            padding=padding, 
                                            truncation=truncation, 
                                            return_tensors='pt')['input_ids']
        return batch_inputs
    
    tokenizer.padding_side = "left"
    remove_columns = dataset.column_names
    dataset = dataset.map(  
                process,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc="Tokenizing dataset...",
                batch_size=batch_size,
                remove_columns= remove_columns
        )
    return dataset
