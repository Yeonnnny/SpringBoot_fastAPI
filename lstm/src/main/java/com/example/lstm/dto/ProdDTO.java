package com.example.lstm.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import org.springframework.web.multipart.MultipartFile;

import com.example.lstm.entity.ProdEntity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@ToString
@Builder
public class ProdDTO {

    private String prodId;
    private String prodName;
    private String prodDesc;
    private int price;
    private String origin;
    private int moq;
    private String unit;
    private ProductCategoryEnum category;
    private LocalDateTime createDate;
    private LocalDateTime updateDate;
    private String remoteIp;
    private String country;
    private int hitCount;
    private double lstmPredictProba;
    private boolean lstmPredict;
    private YesOrNo judge;
    private String customerId;
    private YesOrNo productDelete;

    private MultipartFile uploadFile;
    
    private String originalFileName;
    private String savedFileName;

    private List<Map<String,Object>> prohibit;

    public static ProdDTO toDTO(ProdEntity prodEntity) {
        return ProdDTO.builder()
                .prodId(prodEntity.getProdId())
                .prodName(prodEntity.getProdName())
                .prodDesc(prodEntity.getProdDesc())
                .price(prodEntity.getPrice())
                .origin(prodEntity.getOrigin())
                .moq(prodEntity.getMoq())
                .unit(prodEntity.getUnit())
                .category(prodEntity.getCategory())
                .createDate(prodEntity.getCreateDate())
                .updateDate(prodEntity.getUpdateDate())
                .remoteIp(prodEntity.getRemoteIp())
                .country(prodEntity.getCountry())
                .hitCount(prodEntity.getHitCount())
                .lstmPredictProba(prodEntity.getLstmPredictProba())
                .lstmPredict(prodEntity.isLstmPredict())
                .judge(prodEntity.getJudge())
                .customerId(prodEntity.getCustomerId())
                .productDelete(prodEntity.getProductDelete())
                .originalFileName(prodEntity.getOriginalFileName())
                .savedFileName(prodEntity.getSavedFileName())
                .build();
    }

}
