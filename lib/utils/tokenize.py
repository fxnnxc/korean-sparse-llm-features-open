__all__ = [
    'get_tokenized_dataset',
]


def get_tokenized_dataset(
    dataset,          # HuggingFace Dataset
    tokenizer,        # HuggingFace Tokenizer
    batch_size,       # 배치 처리 크기
    num_proc=None,    # 병렬 처리 프로세스 수
    target_name='text',      # 토큰화할 컬럼 이름
    output_name='input_ids', # 출력 토큰 ID 컬럼 이름
    preserve_names=[],       # 보존할 컬럼들
    max_length=None,         # 최대 시퀀스 길이
    padding=True,            # 패딩 여부
    truncation=False,        # 잘라내기 여부
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
    tokenizer.padding_side = 'left'  # 왼쪽 패딩

    # 6. Dataset 변환
    remove_columns = dataset.column_names  # 기존 컬럼 모두 제거
    dataset = dataset.map(
        process,
        batched=True,           # 배치 처리
        num_proc=num_proc,      # 병렬 처리
        load_from_cache_file=False,  # 캐시 사용 안 함
        desc="Tokenizing dataset...",
        batch_size=batch_size,
        remove_columns=remove_columns,  # 기존 컬럼 제거
    )

    return dataset
