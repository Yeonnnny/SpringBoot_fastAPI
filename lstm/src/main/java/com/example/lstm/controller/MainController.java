package com.example.lstm.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import jakarta.servlet.http.HttpServletRequest;

@Controller
public class MainController {
    @Autowired
    private Environment environment;

    @GetMapping({ "/", "" })
    public String index(HttpServletRequest request, Model model) {
        String forwardedFor = request.getHeader("X-Forwarded-For");
        String realIp;

        // Check if X-Forwarded-For header is present
        if (forwardedFor != null && !forwardedFor.isEmpty()) {
            realIp = forwardedFor.split(",")[0];
        } else {
            // If X-Forwarded-For header is not present, fallback to request.getRemoteAddr()
            realIp = request.getRemoteAddr();
        }

        // Handle IPv6 localhost address
        if (realIp.equals("0:0:0:0:0:0:0:1")) {
            // Get the default IP address from application.properties or fallback to
            // localhost
            realIp = environment.getProperty("server.address", "localhost");
        }

        model.addAttribute("remoteIp", realIp);
        return "index";
    }
}
