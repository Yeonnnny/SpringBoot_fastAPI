package com.example.lstm.entity;

import java.time.LocalDateTime;

import org.hibernate.annotations.UpdateTimestamp;
import org.springframework.data.annotation.CreatedDate;

import com.example.lstm.dto.ProdDTO;
import com.example.lstm.dto.ProductCategoryEnum;
import com.example.lstm.dto.YesOrNo;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

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
@Entity
@Table(name = "product")
public class ProdEntity {

    @Id
    @Column(name = "product_id")
    private String prodId;

    @Column(name = "product_name", nullable = false)
    private String prodName;

    @Column(name = "product_desc", nullable = false)
    private String prodDesc;

    @Column(name = "price", nullable = false)
    private int price;

    @Column(name = "origin", nullable = false)
    private String origin;

    @Column(name = "moq", nullable = false)
    private int moq;

    @Column(name = "unit", nullable = false)
    private String unit;

    @Column(name = "category", nullable = false)
    private ProductCategoryEnum category;

    @Column(name = "create_date")
    @CreatedDate
    private LocalDateTime createDate;

    @Column(name = "update_date")
    @UpdateTimestamp
    private LocalDateTime updateDate;

    @Column(name = "remote_ip", nullable = false)
    private String remoteIp;
    private String country;

    @Column(name = "hit_count")
    private int hitCount;

    @Column(name = "lstm_predict_proba", nullable = false)
    private double lstmPredictProba;

    @Column(name = "lstm_predict", nullable = false)
    private boolean lstmPredict;

    private YesOrNo judge;

    @Column(name = "customer_id", nullable = false)
    private String customerId;

    @Column(name = "product_delete")
    private YesOrNo productDelete;

    @Column(name = "original_file_name")
    private String originalFileName;
    @Column(name = "saved_file_name")
    private String savedFileName;

    public static ProdEntity toEntity(ProdDTO prodDTO) {
        return ProdEntity.builder()
                .prodId(prodDTO.getProdId())
                .prodName(prodDTO.getProdName())
                .prodDesc(prodDTO.getProdDesc())
                .price(prodDTO.getPrice())
                .origin(prodDTO.getOrigin())
                .moq(prodDTO.getMoq())
                .unit(prodDTO.getUnit())
                .category(prodDTO.getCategory())
                .createDate(prodDTO.getCreateDate())
                .updateDate(prodDTO.getUpdateDate())
                .remoteIp(prodDTO.getRemoteIp())
                .country(prodDTO.getCountry())
                .hitCount(prodDTO.getHitCount())
                .lstmPredictProba(prodDTO.getLstmPredictProba())
                .lstmPredict(prodDTO.isLstmPredict())
                .judge(prodDTO.getJudge())
                .customerId(prodDTO.getCustomerId())
                .productDelete(prodDTO.getProductDelete())
                .originalFileName(prodDTO.getOriginalFileName())
                .savedFileName(prodDTO.getSavedFileName())
                .build();
    }

}
