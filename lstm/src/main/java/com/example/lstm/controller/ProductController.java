package com.example.lstm.controller;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.example.lstm.dto.ProdDTO;
import com.example.lstm.service.ModelService;
import com.example.lstm.service.ProductService;


@Controller
@Slf4j
@RequiredArgsConstructor
public class ProductController {
    private final ProductService productService;
    private final ModelService modelService;

    // prodId 생성
    private static int serialNumber = 1;

    private static String getCurrentDateAsString() {
        // 현재 날짜를 yyyyMMdd 형식의 문자열로 변환
        LocalDate currentDate = LocalDate.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
        return currentDate.format(formatter);
    }

    public static String generateId(String prefix) {
        // 두 글자의 prefix와 일련번호, 그리고 현재 날짜를 조합하여 아이디 생성
        String id = prefix + String.format("%05d", serialNumber) + "-" + getCurrentDateAsString();
        serialNumber++; // 일련번호 증가
        return id;
    }

    // 이미지 업로드 경로
    @Value("${spring.servlet.multipart.location}")
    String uploadPath;

    /**
     * 상품 정보 입력받아 DB에 저장 요청
     * 
     * @param prodDTO
     * @param model
     * @return
     */
    @PostMapping("/product/insert")
    public String insert(@ModelAttribute ProdDTO prodDTO, Model model) {

        // prodId 생성
        String categoryCode = prodDTO.getCategory().getCategoryCode();
        String prodId = ProductController.generateId(categoryCode);
        log.info("==========prodId 생성해써 : " + prodId);
        prodDTO.setProdId(prodId);

        // DB에 저장
        ProdDTO dto = productService.insert(prodDTO);

        model.addAttribute("product", dto);
        model.addAttribute("uploadPath", uploadPath); // 이미지 출력을 위해

        return "productDetail";
    }

    @GetMapping("/product/getLstm")
    @ResponseBody
    public List<Map<String, Object>> getLstm(@RequestParam(name = "prodName")String prodName, 
            @RequestParam(name = "prodDesc") String prodDesc) {

        List<Map<String, Object>> result = productService.resultLstm(prodName, prodDesc);
        log.info("{}",result);
        // String data = result.replace("'", "\"");
        // log.info(data);
        return result;
    }
    

    // ---------------------------- IP -------------------------------------
    @GetMapping("/product/getIp")
    public String getIp() {
        return "confirmIP";
    }

}
