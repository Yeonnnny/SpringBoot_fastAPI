package com.example.lstm.service;

import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.example.lstm.dto.ProdDTO;
import com.example.lstm.dto.modelDTO.Ip;
import com.example.lstm.dto.modelDTO.Lstm;
import com.example.lstm.entity.ProdEntity;
import com.example.lstm.repository.ProductRepository;
import com.example.lstm.util.FileService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
@RequiredArgsConstructor
public class ProductService {
    private final ProductRepository repository;
    private final ModelService modelService;

    // 업로드된 파일이 저장될 디렉토리 경로를 읽어옴
    @Value("${spring.servlet.multipart.location}")
    String uploadPath;

    public ProdDTO insert(ProdDTO dto) {
        String originalFileName = null;
        String savedFileName = null;

        // 첨부파일이 있으면 파일명 세팅 실시
        if (!dto.getUploadFile().isEmpty()) {
            originalFileName = dto.getUploadFile().getOriginalFilename();
            savedFileName = FileService.saveFile(dto.getUploadFile(), uploadPath);

            dto.setOriginalFileName(originalFileName);
            dto.setSavedFileName(savedFileName); // entity로 변환 전 dto의 savedFileName 변경해주기
        }

        // lstm
        Lstm lstm = new Lstm(dto.getProdId(), dto.getProdName(), dto.getProdDesc()); // lstm 객체 생성

        Map<String, String> result = modelService.predictLSTM(lstm);
        // {'lstm_predict':lstm_predict,
        // 'lstm_predict_proba':np.round(lstm_predict_proba*100,2)} 이 형식으로 결과 반환됨

        log.info("================= lstm결과 : {}", result);
        // dto.setLstmPredict(Boolean.parseBoolean(result.get("lstm_predict")));
        // dto.setLstmPredictProba(Float.parseFloat(result.get("lstm_predict_proba")));

        return null;

        // ProdEntity entity = ProdEntity.toEntity(dto);
        // repository.save(entity);

        // return ProdDTO.toDTO(entity);

    }
}
