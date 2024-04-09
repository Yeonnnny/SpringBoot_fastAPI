package com.example.lstm.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.example.lstm.entity.ProdEntity;

public interface ProductRepository extends JpaRepository<ProdEntity, String> {
}
