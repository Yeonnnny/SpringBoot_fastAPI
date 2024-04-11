package com.example.lstm.service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.example.lstm.dto.modelDTO.Lstm;


import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class ModelService {
    private final RestTemplate restTemplate;

    // lstm server 경로
    @Value("${lstm.predict.server}")
    String lstmUrl;

    // public List<Map<String, Object>> predictLSTM(Lstm lstm) {
    //     ObjectMapper objectMapper = new ObjectMapper();
    //     HttpHeaders headers = new HttpHeaders();
    //     headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
    //     headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

    //     HttpEntity<Lstm> requestEntity = new HttpEntity<>(lstm, headers);

    //     try {
    //         ResponseEntity<String> response = restTemplate.postForEntity(lstmUrl, requestEntity, String.class);
    //         String json = response.getBody();
    //         log.info(json);
    //         return objectMapper.readValue(json, new TypeReference<List<Map<String, Object>>>() {});
    //     } catch (Exception e) {
    //         e.printStackTrace();
    //         // 오류 처리
    //         return null; // 실제 애플리케이션에서는 더 적절한 오류 처리를 수행해야 합니다.
    //     }
    // }

    public List<Map<String, Object>> predictLSTM(Lstm lstm) {
        // Map<String, Object> error = new HashMap<>();
        List<Map<String, Object>> error = new ArrayList<>();
        List<Map<String, Object>> result = new ArrayList<>();

        try {
            // 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));
            log.info("호호호호호");

            ResponseEntity<List> response = restTemplate.postForEntity(lstmUrl, lstm, List.class);
            log.info("하하하하");
            result = response.getBody();
        } catch (Exception e) {
            Map<String, Object> map = new HashMap<>();
            map.put("statusCode", "450");
            error.add(map);
            return error;
        }
        return result;
    }

}
