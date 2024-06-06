def attach_polarity_score(data, sentiment_polarity):
    # 定义计算情感极性得分的函数
    def calculate_polarity_score(sentiment_labels, sentiment_polarity):
        polarity_score = 0
        for label in sentiment_labels:
            polarity_score += sentiment_polarity.get(label, 0)  # 如果标签不存在于情感极性字典中，则默认为0
        return polarity_score

    # 确定情感标签的列名或列名列表
    sentiment_label_columns = list(sentiment_polarity.keys())

    # 将计算得到的情感极性得分保存到新的列中
    data['polarity_score'] = data[sentiment_label_columns].apply(
        lambda x: calculate_polarity_score(x.index[x.astype(bool)], sentiment_polarity), axis=1)

    return data

