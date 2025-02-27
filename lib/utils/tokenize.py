__all__ = [
    'get_tokenized_dataset',
]


def get_tokenized_dataset(
    dataset,  # HuggingFace Dataset
    tokenizer,  # HuggingFace Tokenizer
    batch_size,
    num_proc=None,  # 병렬 처리 프로세스 수
    target_name='text',
    output_name='input_ids',
    preserve_names=[],
    max_length=None,
    padding=True,
    truncation=False,
):

    # 배치 단위 토큰화 함수
    def process(samples):

        batch_inputs = {}

        # 1. 메인 텍스트 토큰화
        tokenized = tokenizer(
            samples[target_name],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt',  # PyTorch tensor로 반환
        )

        # 2. 토큰화 결과 저장
        batch_inputs[output_name] = tokenized['input_ids']
        batch_inputs['attention_mask'] = tokenized['attention_mask']

        # 3. 보존할 컬럼들 복사
        for v in preserve_names:
            batch_inputs[v] = samples[v]

        # 4. 레이블이 문자열인 경우 토큰화
        if 'label' in samples and isinstance(samples['label'][0], str):
            batch_inputs['label_ids'] = tokenizer(
                samples['label'],
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors='pt',
            )['input_ids']
        return batch_inputs

    # 5. 토크나이저 설정
    tokenizer.padding_side = 'left'

    # 6. Dataset 변환
    remove_columns = dataset.column_names  # 기존 컬럼 모두 제거
    dataset = dataset.map(
        process,
        batched=True,
        num_proc=num_proc,  # 병렬 처리
        load_from_cache_file=False,
        desc="Tokenizing dataset",
        batch_size=batch_size,
        remove_columns=remove_columns,
    )

    return dataset
