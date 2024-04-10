package com.example.lstm.service;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.example.lstm.dto.modelDTO.Ip;
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

    public Map<String, String> predictLSTM(Lstm lstm) {
        Map<String, String> error = new HashMap<>();
        Map<String, String> result = null;

        try {
            // 헤더 설정
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

            ResponseEntity<Map> response = restTemplate.postForEntity(lstmUrl, lstm, Map.class);
            result = response.getBody();
        } catch (Exception e) {
            error.put("statusCode", "450");
            error.put("body", "오류났어요");
            return error;
        }
        return result;
    }

}
