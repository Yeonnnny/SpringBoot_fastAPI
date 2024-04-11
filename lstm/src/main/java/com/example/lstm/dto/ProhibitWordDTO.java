package com.example.lstm.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import com.example.lstm.dto.check.ProhibitReason;


@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@ToString
@Builder
public class ProhibitWordDTO {
    private String prohibitWord;
    private ProhibitReason prohibitReason;

}
