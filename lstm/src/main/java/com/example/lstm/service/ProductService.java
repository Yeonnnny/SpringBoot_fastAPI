package com.example.lstm.service;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.tomcat.util.json.JSONParser;
import org.apache.tomcat.util.json.ParseException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.example.lstm.dto.ProdDTO;
import com.example.lstm.dto.ProhibitSimilarWordDTO;
import com.example.lstm.dto.modelDTO.Lstm;
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
        log.info("============ 파일 저장해써"); 
        // lstm
        Lstm lstm = new Lstm(dto.getProdName(), dto.getProdDesc()); // lstm 객체 생성
        
        List<Map<String, Object>> result = modelService.predictLSTM(lstm);
        log.info("============ {}", result);
        log.info("============ {}", result.get(0).get("lstm_predict"));
        
        Boolean lstmPredict = false;
        Double lstmPredictProba= 0.0;

        
        if (result.size()==1) {
            lstmPredict = String.valueOf(result.get(0).get("lstm_predict")).equals("1")? true:false;
            lstmPredictProba = Double.parseDouble(String.valueOf(result.get(0).get("lstm_predict_proba")));
        }else{
            lstmPredict = String.valueOf(result.get(0).get("lstm_predict")).equals("1")? true:false;
            lstmPredictProba = Double.parseDouble(String.valueOf(result.get(0).get("lstm_predict_proba")));

            result.remove(0);

            // 중복제거
            // LinkedHashSet을 사용하여 중복 제거하면서 순서 유지
            Set<Map<String, Object>> resultSet = new LinkedHashSet<>(result);
            result.clear();
            result.addAll(resultSet);

            for(int i=0;i<result.size();i++){
                
                String similarWord = String.valueOf(result.get(i).get("Similar_Word"));
                String prohibitWord = String.valueOf(result.get(i).get("Prohibited_Word"));
                Double similarProba = Double.parseDouble(String.valueOf(result.get(i).get("Similarity_Score")));
                
                ProhibitSimilarWordDTO prohibitDTO = new ProhibitSimilarWordDTO(1L,similarWord,similarProba,prohibitWord,"p"+i);
                log.info("{}",prohibitDTO);
                // 여기서 prohibitSmilarDB에 save하면 될 듯
            }
        }

        dto.setLstmPredict(lstmPredict);
        dto.setLstmPredictProba(lstmPredictProba);
        log.info("============ dto에 저장해써 : {}", lstmPredict);
        // 여기서 이제 productDB에 save하면 될 듯
        // ProdEntity entity = ProdEntity.toEntity(dto);
        // repository.save(entity);

        // return ProdDTO.toDTO(entity);
        return dto;


    }

    public List<Map<String, Object>> resultLstm(String prodName, String prodDesc){
        // lstm
        Lstm lstm = new Lstm(prodName, prodDesc); // lstm 객체 생성

        List<Map<String, Object>> result = modelService.predictLSTM(lstm);
        // String result = modelService.predictLSTM(lstm);
        return result;
    }

}
