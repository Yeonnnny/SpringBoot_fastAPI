package com.example.lstm.dto;

public enum ProductCategoryEnum {
    FOOD_BEVERAGE("FB"),
    CHEMICAL("CH"),
    HEALTH_MEDICAL("HM"),
    ELECTRONIC("EL"),
    COSMETIC("CO");

    private String categoryCode;

    private ProductCategoryEnum(String categoryCode) {
        this.categoryCode = categoryCode;
    }

    public String getCategoryCode() {
        return categoryCode;
    }

}
