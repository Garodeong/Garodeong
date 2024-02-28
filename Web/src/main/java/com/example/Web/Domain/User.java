package com.example.Web.Domain;

import lombok.*;

import javax.persistence.*;

@Entity
@Table(name = "member")
@ToString
@Builder
@Getter @Setter
@AllArgsConstructor
@NoArgsConstructor
public class User {
    @Id @GeneratedValue
    private Long id;
    private Long deviceId;

    private String name;
    private String loginId;
    private String password;

    //소셜 로그인 시 생성
    private String provider;
    private String providerId;
    private String Email;

    private UserRole role;


}