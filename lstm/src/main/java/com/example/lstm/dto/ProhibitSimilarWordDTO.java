package com.example.lstm.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;


@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@ToString
@Builder
public class ProhibitSimilarWordDTO {
    private Long prohibitSimilarId;
    private String similarWord;
    private double similarProba;
    private String prohibitWord;
    private String productId;



}
